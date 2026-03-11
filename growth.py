import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import scprep
import torch
import time
from utils import seed_everything, load_data, sigmoid_gumbel_sample,get_logger,makedirs,count_parameters
# import atongtf.dataset as atd
import scanpy as sc 
from sklearn.decomposition import PCA
from sinkhorn_knopp_unbalanced import sinkhorn_knopp_unbalanced
def load_data():
    adata = sc.read_h5ad('/media/lenovo/A06B2FA1620B6FCB/LU/retina_data/retina_aligned_pca.h5ad')
    labels = np.array(adata.obs["Time"])
    data = adata.X
    #spatial_coords = adata.obsm['spatial']  
    
    #spatial_coords=spatial_coords
    # if args.whiten:
    #     scaler = StandardScaler()
    #     scaler.fit(data)
    #     data = scaler.transform(data)
    #     if self.velocity is not None:
    #         self.velocity = self.velocity / scaler.scale_
    # self.use_velocity = self.velocity is not None
    emb=adata.obsm['X_pca']
    #pca = PCA(n_components=2)
    #emb_2d = pca.fit_transform(emb)
    
    #scaler = StandardScaler()
    #scaler.fit(emb)
    #transformed = scaler.transform(emb)
    #scaler = StandardScaler()
    #scaler.fit(emb)
    #transformed = scaler.transform(emb)
    #data = np.hstack([data, spatial_coords])
    #max_dim = args.max_dim
    #if max_dim is not None and data.shape[1] > max_dim:
        #print(f"Warning: Clipping dimensionality from {data.shape[1]} to {max_dim}")
        #data = data[:, :max_dim]
        #if args.use_velocity:
            #velocity = velocity[:, :max_dim]
    return emb, labels,

#def load_data_full():
    
    #data = atd.EB_Velocity_Dataset()
    #labels = data.data["sample_labels"]
    #scaler = StandardScaler()
    #scaler.fit(data.emb)
    #transformed = scaler.transform(data.emb)
    #return transformed, labels, scaler


def get_transform_matrix(gamma, a, epsilon=1e-8):
    """Return matrix such that T @ a = b
    gamma : gamma @ 1 = a; gamma^T @ 1 = b
    """
    return (np.diag(1.0 / (a + epsilon)) @ gamma).T


def get_growth_coeffs(gamma, a, epsilon=1e-8, normalize=False):
    T = get_transform_matrix(gamma, a, epsilon)
    unnormalized_coeffs = np.sum(T, axis=0)
    if not normalize:
        return unnormalized_coeffs
    return unnormalized_coeffs / np.sum(unnormalized_coeffs) * len(unnormalized_coeffs)


#data, labels, _ = load_data_full()
data, labels = load_data()
print(data.shape, labels.shape)
#exit()

# Compute couplings

timepoints = np.unique(labels)

dfs = [data[labels == tp] for tp in timepoints]
#pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4),(4,5),(3,5)]

def get_all_growth_coeffs(alpha):
    gcs = []
    for i in range(len(dfs) - 1):
        a, b = dfs[i], dfs[i + 1]
        m, n = a.shape[0], b.shape[0]
        M = cdist(a, b)
        entropy_reg = 0.9
        reg_1, reg_2 = alpha, 10000
        gamma = sinkhorn_knopp_unbalanced(
            np.ones(m) / m, np.ones(n) / n, M, entropy_reg, reg_1, reg_2
        )
        gc = get_growth_coeffs(gamma, np.ones(m) / m)
        gcs.append(gc)
    return gcs
gcs = get_all_growth_coeffs(1.5)
gcs = np.concatenate(gcs)
print(gcs)
np.save('gcs_retina.npy', gcs)
def evaluate_alpha(alpha):
    gcs = get_all_growth_coeffs(alpha)
    gcs = np.concatenate(gcs)
    return {
        "alpha": alpha,
        "mean": np.mean(gcs),
        "std": np.std(gcs),
        "max": np.max(gcs),
        "min": np.min(gcs),
        "outlier_ratio": np.sum((gcs > np.mean(gcs) + 3*np.std(gcs)) | (gcs < np.mean(gcs) - 3*np.std(gcs))) / len(gcs)
    }

# 测试所有候选值
results = evaluate_alpha(1.5) 
print(results)
#gcs = np.load("gcs.npy")
#print(gcs)
device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class GrowthNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(51, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


X = np.concatenate([data, labels[:, None]], axis=1)[labels != timepoints[-1]]
Y = gcs[:, None]

#device = torch.device("cuda:" + str(1) if torch.cuda.is_available() else "cpu")

model = GrowthNet().to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters())

for it in range(10000):
    optimizer.zero_grad()
    batch_idx = np.random.randint(len(X), size=256)
    x = torch.from_numpy(X[batch_idx,:]).type(torch.float32).to(device)
    y = torch.from_numpy(Y[batch_idx,:]).type(torch.float32).to(device)
    negative_samples = np.concatenate([np.random.uniform(size=(256,50)) * 8 - 4,
                                       np.random.choice(timepoints, size=(256,1))], axis=1)
    negative_samples = torch.from_numpy(negative_samples).type(torch.float32).to(device)
    x = torch.cat([x, negative_samples])
    y = torch.cat([y, torch.ones_like(y)])
    pred = model(x)
    loss = torch.nn.MSELoss()
    output = loss(pred, y)
    output.backward()
    optimizer.step()
    if it % 100 == 0:
        print(it, output)

#torch.save(model, 'model')
torch.save(model, '/media/lenovo/A06B2FA1620B6FCB/LU/retina_data/grow_model_1.5.pth')

model.eval()
import matplotlib

for i, tp in enumerate(np.linspace(0, 51, 100)):
    fig, axes = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [7, 1]}, figsize=(8, 8)
    )
    ax = axes[0]
    npts = 200
    side = np.linspace(-4, 4, npts)
    xx, yy = np.meshgrid(side, side)
    xx = torch.from_numpy(xx).type(torch.float32).to(device)
    yy = torch.from_numpy(yy).type(torch.float32).to(device)
    z_grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)
    data_in = torch.cat([z_grid, tp * torch.ones(z_grid.shape[0], 1).to(device)], 1)
    gr = model(data_in)
    gr = gr.reshape(npts, npts).to("cpu").detach().numpy()
    ax.pcolormesh(
        xx.cpu().detach(), yy.cpu().detach(), gr, cmap="RdBu_r", vmin=0, vmax=2
    )
    scprep.plot.scatter2d(data, c="Gray", ax=ax, alpha=0.5)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title("Growth Rate")
    # ax.set_title("%0.2f" % tp, fontsize=32)
    ax = axes[1]
    # Colorbar
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap="Spectral", orientation="horizontal")
    cb.set_ticks(np.linspace(0, 1, 4))
    cb.set_ticklabels(np.arange(4))
    ax.axvline(tp / 3, c="k", linewidth=15)
    ax.set_title("Time")
    plt.savefig("data/viz-%05d.jpg" % i)
    plt.close()