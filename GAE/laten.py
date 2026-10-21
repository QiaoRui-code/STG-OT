import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


# =========================
# Reproducibility
# =========================
def seed_all(seed=19491001):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# GAE modules
# =========================
class content_graph_conv(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float32))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros((out_features,), dtype=torch.float32))
        else:
            self.register_parameter("bias", None)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = x @ self.weight
        if self.bias is not None:
            x = x + self.bias
        return torch.mm(adj, x)


class Encoder(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: list, dim_latent: int):
        super().__init__()
        self.net = self._build_layers(dim_in, dim_hidden, dim_latent)

    def _build_layers(self, dim_in, dim_hidden, dim_out):
        if len(dim_hidden) == 0:
            return nn.Sequential(nn.Linear(dim_in, dim_out))
        return nn.Sequential(
            nn.Linear(dim_in, dim_hidden[0]),
            nn.ReLU(),
            self._build_layers(dim_hidden[0], dim_hidden[1:], dim_out),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: list, dim_out: int):
        super().__init__()
        self.net = self._build_layers(dim_in, dim_hidden, dim_out)

    def _build_layers(self, dim_in, dim_hidden, dim_out):
        if len(dim_hidden) == 0:
            return nn.Sequential(nn.Linear(dim_in, dim_out))
        return nn.Sequential(
            nn.Linear(dim_in, dim_hidden[0]),
            nn.ReLU(),
            self._build_layers(dim_hidden[0], dim_hidden[1:], dim_out),
        )

    def forward(self, x):
        return self.net(x)


class GAE(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_latent, in_logarithm=True):
        super().__init__()
        self.gcn_layer = content_graph_conv(dim_in, dim_hidden[0])
        self.encoder = Encoder(dim_hidden[0], dim_hidden[1:], dim_latent)
        self.decoder = Decoder(dim_latent, dim_hidden[::-1], dim_in)
        self.in_logarithm = in_logarithm
        self.bn = nn.BatchNorm1d(dim_hidden[0])

    def forward(self, x, adj, idx):
        if self.in_logarithm:
            x = torch.log1p(x)
        h = F.relu(self.bn(self.gcn_layer(x, adj)))
        h = h[idx, :]
        z = self.encoder(h)
        out = self.decoder(z)
        return out, z

    def forward_loss(self, x, adj, idx):
        y, z = self.forward(x, adj, idx)
        target = torch.log1p(x[idx, :]) if self.in_logarithm else x[idx, :]
        loss = F.mse_loss(y, target)
        return loss, y, z

    @torch.no_grad()
    def get_latent_representation(self, x, adj, idx):
        _, z = self.forward(x, adj, idx)
        return z


# =========================
# Graph helper
# =========================
def normalize_adj_binary_knn(adj_binary):
    A = ((adj_binary + adj_binary.T) > 0).astype(np.float32)
    np.fill_diagonal(A, 1.0)
    deg = A.sum(axis=1)
    deg[deg == 0] = 1.0
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    A_norm = d_inv_sqrt @ A @ d_inv_sqrt
    return A_norm.astype(np.float32)


# =========================
# Train and write latent
# =========================
def create_latent_space_and_update_adata(
    adata,
    dim_latent=32,
    model_save_path=None,
    n_epochs=200,
    n_neighbors=5,
    lr=1e-3,
    device=None,
    use_me_regularization=True,
    me_key="X_me",
    lambda_me=1e-3,
    me_warmup=50,
    grad_clip=1.0,
):
    """
    Output:
      updated_adata.obsm['X_latent']      -> raw latent
      updated_adata.obsm['X_latent_01']   -> sigmoid-compressed latent in [0,1]
      updated_adata.obsm['spatial_norm']
      updated_adata.obsm['spatial_original']
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) expression
    expression_data = adata.X
    if sp.issparse(expression_data):
        expression_data = expression_data.toarray()
    expression_data = expression_data.astype(np.float32)

    # auto choose log1p usage
    in_logarithm = bool(np.min(expression_data) >= 0.0)

    # 2) spatial -> graph
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] not found.")
    spatial_coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    coord_scaler = StandardScaler()
    coords_normalized = coord_scaler.fit_transform(spatial_coords).astype(np.float32)

    adj_binary = kneighbors_graph(
        coords_normalized,
        n_neighbors=n_neighbors,
        mode="connectivity",
        include_self=True,
    ).toarray().astype(np.float32)
    adj_norm = normalize_adj_binary_knn(adj_binary)

    # 3) tensors
    X = torch.tensor(expression_data, dtype=torch.float32, device=device)
    adj = torch.tensor(adj_norm, dtype=torch.float32, device=device)
    idx = torch.arange(X.shape[0], device=device)

    # 4) model
    n_genes = expression_data.shape[1]
    model = GAE(
        dim_in=n_genes,
        dim_hidden=[256, 128],
        dim_latent=dim_latent,
        in_logarithm=in_logarithm,
    ).to(device)

    # 5) optional ME head: z -> X_me
    me_head = None
    ME = None
    if use_me_regularization:
        if me_key not in adata.obsm:
            raise KeyError(f"adata.obsm['{me_key}'] not found.")
        ME_np = np.asarray(adata.obsm[me_key], dtype=np.float32)
        if ME_np.shape[0] != X.shape[0]:
            raise ValueError(f"{me_key} rows must match n_cells: {ME_np.shape[0]} vs {X.shape[0]}")
        ME = torch.tensor(ME_np, dtype=torch.float32, device=device)
        me_head = nn.Linear(dim_latent, ME.shape[1]).to(device)

    params = list(model.parameters()) + (list(me_head.parameters()) if me_head is not None else [])
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)

    # 6) train
    model.train()
    if me_head is not None:
        me_head.train()

    print("Start training GAE...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        recon_loss, _, z = model.forward_loss(X, adj, idx)

        if me_head is not None:
            me_hat = me_head(z)
            me_loss = F.mse_loss(me_hat, ME[idx])
            if epoch < me_warmup:
                total_loss = recon_loss
                cur_lambda = 0.0
            else:
                total_loss = recon_loss + lambda_me * me_loss
                cur_lambda = lambda_me
        else:
            me_loss = torch.tensor(0.0, device=device)
            total_loss = recon_loss
            cur_lambda = 0.0

        total_loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"Total {total_loss.item():.6f} | "
                f"Recon {recon_loss.item():.6f} | "
                f"ME {me_loss.item():.6f} | "
                f"lambda {cur_lambda:.1e}"
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 7) save checkpoint
    if model_save_path is not None:
        save_dir = os.path.dirname(model_save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        save_obj = {
            "model_state_dict": model.state_dict(),
            "dim_in": n_genes,
            "dim_hidden": [256, 128],
            "dim_latent": dim_latent,
            "in_logarithm": in_logarithm,
            "n_neighbors": n_neighbors,
            "use_me_regularization": use_me_regularization,
            "me_key": me_key,
            "lambda_me": float(lambda_me),
            "me_warmup": int(me_warmup),
            "decoder_output_space": "log1p_expression" if in_logarithm else "expression",
        }
        if me_head is not None:
            save_obj["me_head_state_dict"] = me_head.state_dict()
            save_obj["me_dim"] = int(ME.shape[1])

        torch.save(save_obj, model_save_path)
        print(f"Model saved to: {model_save_path}")

    # 8) latent
    model.eval()
    latent = model.get_latent_representation(X, adj, idx).detach().cpu().numpy()

    # 9) update AnnData
    updated_adata = adata.copy()
    updated_adata.obsm["X_gae"] = latent
    updated_adata.obsm["X_latent_01"] = 1.0 / (1.0 + np.exp(-latent))  # compare-friendly bounded latent
    updated_adata.obsm["spatial_norm"] = coords_normalized
    updated_adata.obsm["spatial_original"] = spatial_coords.copy()

    if me_head is not None:
        me_head.eval()
        with torch.no_grad():
            _, z = model.forward(X, adj, idx)
            me_pred = me_head(z).detach().cpu().numpy()
        updated_adata.obsm["X_me_pred_from_latent"] = me_pred

    updated_adata.uns["gae_info"] = {
        "dim_latent": dim_latent,
        "n_neighbors": n_neighbors,
        "n_epochs": n_epochs,
        "coord_scaler_mean": coord_scaler.mean_.tolist(),
        "coord_scaler_scale": coord_scaler.scale_.tolist(),
        "model_save_path": model_save_path,
        "in_logarithm": in_logarithm,
        "use_me_regularization": use_me_regularization,
        "me_key": me_key,
        "lambda_me": float(lambda_me),
        "me_warmup": int(me_warmup),
        "grad_clip": float(grad_clip) if grad_clip is not None else None,
    }

    return updated_adata


# =========================
# Load checkpoint and decode
# =========================
def load_gae_from_checkpoint(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device)

    model = GAE(
        dim_in=ckpt["dim_in"],
        dim_hidden=ckpt["dim_hidden"],
        dim_latent=ckpt["dim_latent"],
        in_logarithm=ckpt["in_logarithm"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    me_head = None
    if "me_head_state_dict" in ckpt:
        me_head = nn.Linear(ckpt["dim_latent"], ckpt["me_dim"]).to(device)
        me_head.load_state_dict(ckpt["me_head_state_dict"], strict=True)
        me_head.eval()

    return model, me_head, ckpt


@torch.no_grad()
def decode_latent_to_metagene(z_np, model, ckpt, device=None):
    """
    z_np -> decoder output space and expression space
    returns:
      x_decoded: decoder output (log1p_expression if in_logarithm=True)
      x_expression: expression space (expm1 if needed)
    """
    if device is None:
        device = next(model.parameters()).device

    z_t = torch.tensor(z_np, dtype=torch.float32, device=device)
    x_decoded = model.decoder(z_t).detach().cpu().numpy()

    if ckpt["in_logarithm"]:
        x_expression = np.expm1(x_decoded)
        x_expression = np.clip(x_expression, 0, None)
    else:
        x_expression = x_decoded.copy()

    return x_decoded, x_expression


# =========================
# Example
# =========================
if __name__ == "__main__":
    seed_all(19491001)

    input_path = "./datasets/Dwith_ME.h5ad"
    model_save_path = "./datasets/Dae_model.pth"
    output_path = "./datasets/Dwith_GAE_latent.h5ad"


    adata = sc.read_h5ad(input_path)

    updated_adata = create_latent_space_and_update_adata(
        adata=adata,
        dim_latent=32,
        model_save_path=model_save_path,
        n_epochs=500,
        n_neighbors=5,
        lr=1e-3,
        use_me_regularization=True,   # requires adata.obsm["X_me"]
        me_key="X_me",
        lambda_me=1e-3,
        me_warmup=50,
        grad_clip=1.0,
    )

    print("Original shape:", adata.shape)
    print("Latent shape:", updated_adata.obsm["X_gae"].shape)
    print("Latent_01 shape:", updated_adata.obsm["X_latent_01"].shape)
    print("X unchanged shape:", updated_adata.X.shape)

    # Optional: decode latent back to metagene/expression space
    model_loaded, me_head_loaded, ckpt = load_gae_from_checkpoint(model_save_path)
    #x_decoded, x_expression = decode_latent_to_metagene(
    #    updated_adata.obsm["X_gae"][:1000], model_loaded, ckpt
   # )
    #print("Decoded shape:", x_decoded.shape, "Expression shape:", x_expression.shape)

    updated_adata.write_h5ad(output_path)
    print(f"Saved updated AnnData to: {output_path}")


