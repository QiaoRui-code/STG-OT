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
    adata = sc.read_h5ad('./sim_dataset/sim_dynamices_aligned.h5ad')
    labels = np.array(adata.obs["time_point"])
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
    #emb=adata.obsm['X_pca']
    emb=adata.X
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
gcs = get_all_growth_coeffs(1.0)
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


results = evaluate_alpha(1.0) 
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
torch.save(model, './sim_datset/grow_model_1.0.pth')

