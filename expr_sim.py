import os
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import scvelo as scv
from sklearn.decomposition import PCA
from data_simulate import load_data_lorenz_96, load_data_simulation, plot_integrate
from utils import seed_everything, load_data, sigmoid_gumbel_sample,get_logger,makedirs,count_parameters
from parse import parser
from model import build_model_tabular,set_cnf_options
from lib.regularization import create_regularization_fns
from train import pred_train,graph_discov,train_eval
from visualize import plot_output, plot_causal_matrix,visualize
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import torch
import numpy as npbuild_knn_graph
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
from sklearn.neighbors import kneighbors_graph
# Set random seed for reproducibility
seed = 19491001
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def compute_granger_matrix(data, maxlag=1, significance=0.05):
  
    n_features = data.shape[1]
    granger_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                continue
          
            test_data = data[:, [i, j]]
            try:
                results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
                p_values = [results[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag+1)]
                min_p = min(p_values)
               
                granger_matrix[i, j] = min(1.0, -np.log10(min_p) / 10.0)
            except:
                granger_matrix[i, j] = 0.0
    return granger_matrix
def load_P_from_me_head(gae_ckpt_path, nonneg_norm=True):
    ckpt = torch.load(gae_ckpt_path, map_location="cpu")
    if "me_head_state_dict" not in ckpt:
        raise KeyError(f"No 'me_head_state_dict' in {gae_ckpt_path}")

    W = ckpt["me_head_state_dict"]["weight"].cpu().numpy()  # (K, d)

    if nonneg_norm:
        P = np.abs(W)
        P = P / (P.sum(axis=1, keepdims=True) + 1e-8)
    else:
        P = W
    return P.astype(np.float32)


def _to_numpy(x):
    import numpy as np
    import torch
    if isinstance(x, torch.nn.Parameter):
        return x.detach().cpu().numpy()
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def project_graph_to_module(A_latent, P, symmetrize=True):
    A_latent = _to_numpy(A_latent).astype(np.float32)
    P = _to_numpy(P).astype(np.float32)
    A_module = P @ A_latent @ P.T
    if symmetrize:
        A_module = 0.5 * (A_module + A_module.T)
    return A_module


def plot_heatmap(A, title, save_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(A, cmap="coolwarm", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def top_changed_module_edges(A_before, A_after, module_names, top_k=100):
    D = A_after - A_before
    rows = []
    K = D.shape[0]
    for i in range(K):
        for j in range(i + 1, K):
            rows.append((i, j, float(D[i, j]), float(abs(D[i, j]))))
    rows = sorted(rows, key=lambda x: x[3], reverse=True)[:top_k]

    df = pd.DataFrame({
        "src_idx": [r[0] for r in rows],
        "dst_idx": [r[1] for r in rows],
        "src_name": [module_names[r[0]] for r in rows],
        "dst_name": [module_names[r[1]] for r in rows],
        "delta_weight": [r[2] for r in rows],
        "abs_delta": [r[3] for r in rows],
    })
    return df
def save_graph_interpret_artifacts(save_dir, A_init, A_opt, P, module_names):
    import os
    import numpy as np
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)

    # 关键：先统一转 numpy（detach）
    A_init = _to_numpy(A_init).astype(np.float32)
    A_opt = _to_numpy(A_opt).astype(np.float32)
    P = _to_numpy(P).astype(np.float32)

    A_module_init = project_graph_to_module(A_init, P, symmetrize=True)
    A_module_opt = project_graph_to_module(A_opt, P, symmetrize=True)
    A_module_delta = A_module_opt - A_module_init

    # 保存
    np.save(os.path.join(save_dir, "A_latent_init.npy"), A_init)
    np.save(os.path.join(save_dir, "A_latent_opt.npy"), A_opt)
    np.save(os.path.join(save_dir, "P_latent_to_module.npy"), P)
    np.save(os.path.join(save_dir, "A_module_init.npy"), A_module_init)
    np.save(os.path.join(save_dir, "A_module_opt.npy"), A_module_opt)
    np.save(os.path.join(save_dir, "A_module_delta.npy"), A_module_delta)

    # top changed edges
    rows = []
    K = A_module_delta.shape[0]
    for i in range(K):
        for j in range(i + 1, K):
            rows.append((i, j, float(A_module_delta[i, j]), abs(float(A_module_delta[i, j]))))
    rows = sorted(rows, key=lambda x: x[3], reverse=True)[:120]

    if module_names is None:
        module_names = [f"ME_{i}" for i in range(K)]

    df = pd.DataFrame({
        "src_idx": [r[0] for r in rows],
        "dst_idx": [r[1] for r in rows],
        "src_name": [module_names[r[0]] for r in rows],
        "dst_name": [module_names[r[1]] for r in rows],
        "delta_weight": [r[2] for r in rows],
        "abs_delta": [r[3] for r in rows],
    })
    df.to_csv(os.path.join(save_dir, "top_changed_module_edgesout.csv"), index=False,float_format="%.8e")

    print(f"[Done] saved to: {save_dir}")
def integrate_backwards(end_samples, model, savedir, ntimes=100, memory=0.1, device="cpu"):
    """Integrate some samples backwards and save the results."""
    with torch.no_grad():
        z = torch.from_numpy(end_samples).type(torch.float32).to(device)
        zero = torch.zeros(z.shape[0], 1).to(z)
        cnf = model.chain[0]

        zs = [z]
        deltas = []
        int_tps = np.linspace(args.int_tps[1], args.int_tps[2], ntimes)
        # for i, itp in enumerate(int_tps[::-1][:-1]):
        for i, itp in enumerate(int_tps[:-1]):
            # tp counts down from last
            timescale = int_tps[1] - int_tps[0]
            integration_times = torch.tensor([itp - timescale, itp])
            # integration_times = torch.tensor([np.linspace(itp - timescale, itp, ntimes)])
            integration_times = integration_times.type(torch.float32).to(device)

            # transform to previous timepoint
            z, delta_logp = cnf(zs[-1], zero, integration_times=integration_times)
            zs.append(z)
            deltas.append(delta_logp)
        zs = torch.stack(zs, 0)
        z_pred = zs[-1]
        zs = zs.cpu().numpy()
        np.save(os.path.join(savedir, "backward_trajectories.npy"), zs)
        return z_pred
def get_model_graph_param(model, graph_shape):
   
    for n, p in model.named_parameters():
        if ("graph" in n.lower() or "adj" in n.lower()) and tuple(p.shape) == tuple(graph_shape):
            print("[graph param]", n, p.shape)
            return p

    for n, p in model.named_parameters():
        if tuple(p.shape) == tuple(graph_shape):
            print("[graph param by shape]", n, p.shape)
            return p
    raise RuntimeError(f"Cannot find graph param in model with shape={graph_shape}")
def main(args, datapath):
    seed_everything(1)
    makedirs(args.save)
    logger = get_logger(
        logpath=os.path.join(args.save, "logs"),
        filepath=os.path.abspath(__file__),
        displaying=~args.no_display_loss,
    )
    logger.info(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adata, data, labels, velocity,spatial_coords = load_data(args, datapath)
    adata.write_h5ad("./sim_dataset/sim_processed_adata_with_scaler.h5ad") 

    # 数据标准化（可选）
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)

    all_timepoints = np.sort(np.unique(adata.obs['time_point']))
    #train_timepoints = all_timepoints[0:5]  # 前5个训练
    #held_out_idx = 1   # 第二个时间点
    #held_out_timepoint = all_timepoints[held_out_idx]
    #train_timepoints = np.delete(all_timepoints, held_out_idx)
     # 删除该索引
    #train_indices = np.isin(labels, train_timepoints)
    #data_train = data[train_indices]
    #labels_train = labels[train_indices]
    args.timepoints = all_timepoints
    args.timepoints_train = all_timepoints
    args.max = max(all_timepoints)
    #args.time_scale = float(args.time_scale)
    args.int_tps = all_timepoints * args.time_scale 
    #args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale
    #args.int_tps = (train_timepoints + 1) * args.time_scale
    # ===== 初始化图（格兰杰因果） =====
    #A_init_np = build_knn_graph(data, k=10, metric='euclidean', mode='connectivity')
    #A_init_np = np.eye(data.shape[1])
    #A_init_np=np.corrcoef(data.T)
    A_init_np = compute_granger_matrix(data, maxlag=2)
    np.fill_diagonal(A_init_np, 0)
    A_init_np = np.clip(A_init_np, 0, 1)
    graph_causal_fit = nn.Parameter(
        torch.tensor(A_init_np, dtype=torch.float32, device=device)
    )
    
    regularization_fns, regularization_coeffs = create_regularization_fns(args)

   
    model = build_model_tabular(
        args, data.shape[1], graph_causal_fit, regularization_fns
    ).to(device)
    set_cnf_options(args, model)
    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

   
    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr)
    graph_param = get_model_graph_param(model, graph_causal_fit.shape)
    optimizer_graph = torch.optim.Adam([graph_param], lr=getattr(args, 'graph_lr', 5e-5))

    
    scheduler_model = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_model, patience=20, factor=0.5)
    scheduler_graph = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_graph, patience=10, factor=0.5)
    #P = load_P_from_me_head('/mnt/sda2/stVCR-m/datasets/Rae_model.pth', nonneg_norm=True)
    #try:
        
        #P_tensor = torch.tensor(P, dtype=torch.float32).to(device)
        #logger.info(f"loading P，shape: {P_tensor.shape}")
    #except Exception as e:
        #logger.info(f"Warning: error: {e}")
        #P_tensor = None
        #P_np = None
    
    #best_val_loss = float('inf')
    #patience = 0
    #early_stop_patience = getattr(args, 'early_stop_patience', 10)

  
    for itr in range(1, args.total_niters + 1):
        logger.info(f"\n========== Outer Iteration {itr} ==========")

      
        model, current_val_loss, best_val_loss = pred_train(
            device, args, data,spatial_coords, graph_causal_fit, labels, velocity,
            model, regularization_coeffs, regularization_fns, logger
        )

      
        graph_causal_fit = graph_discov(
            device, args, data,spatial_coords, graph_causal_fit, labels, velocity,
            model, optimizer_graph, regularization_coeffs, regularization_fns, logger,P_tensor=graph_causal_fit
        )
        
        
        scheduler_model.step(current_val_loss)
        scheduler_graph.step(current_val_loss)

       
        torch.save(model, "./sim_dataset/sim_graph_checkpt.pth",)
        torch.save(graph_causal_fit , "./sim_dataset/sim_causal_fit.pth", )
       
        if itr % 10 == 0:
            lr_m = optimizer_model.param_groups[0]['lr']
            lr_g = optimizer_graph.param_groups[0]['lr']
            logger.info(f"LR: model={lr_m:.2e}, graph={lr_g:.2e}")

     
        if itr % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(args.save, f"model_{itr}.pth"))
            torch.save(graph_causal_fit, os.path.join(args.save, f"graph_{itr}.pth"))

    
    best_model_path = os.path.join(args.save, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        best_graph_path = os.path.join(args.save, "best_graph.pth")
        if os.path.exists(best_graph_path):
            graph_causal_fit.data = torch.load(best_graph_path).data
        logger.info("Loaded best model and graph.")
   
    graph_param = get_model_graph_param(model, (3, 3))

    A_causal = torch.sigmoid(graph_param).detach().cpu().numpy().astype(np.float32)
    np.fill_diagonal(A_causal, 0.0)  


    self_dyn = model.chain[0].odefunc.odefunc.diffeq.layers[0].self_dyn.detach().cpu().numpy().astype(np.float32)


    np.save(os.path.join(args.save, "A_causal_off_sim.npy"), A_causal)
    np.save(os.path.join(args.save, "self_dyn_sim.npy"), self_dyn)


    #if "me_names" in adata.uns:
        #module_names = list(adata.uns["me_names"])
    #else:
        #K = P.shape[0]
        #module_names = [f"ME_{i}" for i in range(K)]


    #save_graph_interpret_artifacts(
    #save_dir=os.path.join(args.save, "graph_interpret_final"),
    #A_init=A_init_np,
    #A_opt=A_causal,
    #P=P,
    #module_names=module_names
#)
    A_opt = A_causal.astype(np.float32)               
    A_init = A_init_np.astype(np.float32)

    print("A_init mean/std:", A_init.mean(), A_init.std())
    print("A_opt  mean/std:", A_opt.mean(), A_opt.std())

    delta = A_opt - A_init
    print("delta latent mean/max:", np.mean(np.abs(delta)), np.max(np.abs(delta)))


    causal_fit = torch.from_numpy(A_opt).to(device)

    #P = load_P_from_me_head('/mnt/sda2/stVCR-m/datasets/Rae_model.pth', nonneg_norm=True)
    #print("P shape:", P.shape)
    #print("P row sums min/max:", P.sum(axis=1).min(), P.sum(axis=1).max())
    #print("P nonzero ratio:", (np.abs(P) > 1e-12).mean())
    #print("P min/max:", P.min(), P.max())

    return data, labels, causal_fit

def eval(args, data, labels, graph_causal_fit):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx = np.arange(data.shape[0])[labels == np.unique(labels)[-1]]
    data_train = np.delete(data, idx, axis=0)
    labels_train = np.delete(labels, idx, axis=0)
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, data.shape[1], graph_causal_fit, regularization_fns, ).to(device)

    # Use maximum timepoint to establish integration_times
    # as some timepoints may be left out for validation etc.

    # state_dict = torch.load("/media/lenovo/6ED3FFE79A41910F/Lu/causal_1105/results/tmp/embry/dim10/graph_checkpt.pth", map_location=device)
    # model.load_state_dict(state_dict["state_dict"])
    #model = torch.load("/media/lenovo/A06B2FA1620B6FCB/causal_1105/results/tmp/embry4/graph_checkpt.pth")
    # direction = model.chain[0].odefunc.odefunc.diffeq(itp, data)

    set_cnf_options(args, model)
    args.data = data
    # args.timepoints = args.data.get_unique_times()
    # args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale

    # end_time_data = data.data_dict[args.embedding_name]
    # end_time_data = data.get_data()[
    #     args.data.get_times() == np.max(args.data.get_times())
    # ]
    # np.random.permutation(end_time_data)
    # rand_idx = np.random.randint(end_time_data.shape[0], size=5000)
    # end_time_data = end_time_data[rand_idx,:]
    end_time_data = data_train[labels_train == np.max(labels_train)]
    integrate_backwards(end_time_data, model, args.save, ntimes=100, device=device)
    #print("integrating backwards")
    
    
    # end_time_data = data[labels == np.min(labels)]
    # end_time_data = data_train[labels_train == np.max(labels_train)]
    #z_pred = integrate_backwards(end_time_data, model, args.save, ntimes=10, device=device)
    #np.save(os.path.join(args.save, "predicted_data.npy"), z_pred.cpu().numpy())
    # data[idx] = z_pred.cpu()
    # savedir = args.save
    #return z_pred

if __name__ == "__main__":
    args = parser.parse_args()
    # main(args,datapath='/media/lenovo/6ED3FFE79A41910F/Lu/causal_1105/results/adata.h5ad')
    #datapath='/mnt/sda2/stVCR-m/datasets/Dwith_GAE_latent3.h5ad'
    datapath='./sim_datset/sim_dynamic_aligned.h5ad'
    data, labels, causal_fit= main(args,datapath)
    #np.save(os.path.join(args.save, "predicted_data.npy"), z_pred.cpu().numpy())
    # scv.pl.velocity_embedding_stream(adata, basis="umap", color="celltype")
    # direction = eval(args, data, labels, causal_fit)
    eval(args, data, labels, causal_fit)
    plot_causal_matrix(causal_fit.cpu().detach().numpy(), figsize=[10, 8], vmin=0, vmax=1)

    # plt.savefig(pred_cg, os.path.join(args.save, "pred_cm.jpg"), dpi=300)
