import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import linear_sum_assignment
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. Utility Functions
# ==============================================================================
def to_numpy(x):
    if x is None: return None
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray): return x
    return np.array(x)

def disperse_points(points, min_dist=0.01, max_iter=100):
    """Physical repulsion algorithm to prevent overlapping points"""
    points = points.copy()
    for _ in range(max_iter):
        dists = squareform(pdist(points))
        np.fill_diagonal(dists, np.inf)
        too_close = np.where(dists < min_dist)
        if len(too_close[0]) == 0: break
        move = np.zeros_like(points)
        for i, j in zip(*too_close):
            vec = points[i] - points[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-10: vec = np.random.randn(2) * 1e-10
            overlap = min_dist - dist
            force = (vec / (dist + 1e-10)) * overlap * 0.5
            move[i] += force
        points += move
    return points

def optimal_transport_match(coords1, coords2, max_points=2000):
    """Optimal transport matching to find point-to-point correspondences"""
    n1, n2 = len(coords1), len(coords2)
    if n1 > max_points: idx1 = np.random.choice(n1, max_points, replace=False)
    else: idx1 = np.arange(n1)
    if n2 > max_points: idx2 = np.random.choice(n2, max_points, replace=False)
    else: idx2 = np.arange(n2)
    cost = cdist(coords1[idx1], coords2[idx2])
    row_ind, col_ind = linear_sum_assignment(cost)
    return idx1[row_ind], idx2[col_ind]

# ==============================================================================
# 1. Custom Feature-Fused FGW
# ==============================================================================
def compute_fgw_loss(M, C1, C2, h2, h3, alpha=0.5, epsilon=0.05, max_iter=20):
    """
    Custom PyTorch implementation of Fused Gromov-Wasserstein (FGW) loss.
    Uses Sinkhorn algorithm, fully differentiable and runs directly on GPU.
    """
    T = torch.outer(h2, h3) # Initialize transport matrix
    C1_sq = C1**2
    C2_sq = C2**2
    
    for _ in range(max_iter):
        # Calculate structural tensor L
        L1 = C1_sq @ h2.unsqueeze(1)
        L2 = (C2_sq @ h3.unsqueeze(1)).T
        L3 = 2 * C1 @ T @ C2.T
        L = L1 + L2 - L3
        
        # Fuse feature matrix M and structural matrix L
        C = (1 - alpha) * M + alpha * L
        
        # Sinkhorn normalization
        K = torch.exp(-C / epsilon)
        
        u = h2 / (K.sum(dim=1) + 1e-9)
        v = h3 / (K.T @ u + 1e-9)
        T = u.unsqueeze(1) * K * v.unsqueeze(0)
        
    return torch.sum(C * T)

# ==============================================================================
# 2. Spatial ODE Network
# ==============================================================================
class SpatialODENet(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256):
        super().__init__()
        # Input dimension = 2(coords) + latent_dim(expression) + 1(time)
        input_dim = 2 + latent_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2) # Output dx/dt, dy/dt
        )
        nn.init.normal_(self.net[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.net[-1].bias, 0)
    
    def forward(self, t, state):
        # state shape: [batch, 2 + latent_dim]
        batch_size = state.shape[0]
        t_expanded = t * torch.ones(batch_size, 1).to(state.device)
        inputs = torch.cat([state, t_expanded], dim=1)
        
        # Network computes velocity
        velocity = self.net(inputs)
        
        # Gene latent variables are assumed to be static
        dz_dt = torch.zeros_like(state[:, 2:]) 
        
        # Concatenate into: [dx/dt, dy/dt, 0, ..., 0]
        return torch.cat([velocity, dz_dt], dim=1)

class ContinuousTimeODEfunc(nn.Module):
    def __init__(self, diffeq):
        super().__init__()
        self.diffeq = diffeq
    def forward(self, t, y):
        return self.diffeq(t, y)

class ContinuousTimeCNF(nn.Module):
    def __init__(self, odefunc, solver='dopri5'):
        super().__init__()
        self.odefunc = odefunc
        self.solver = solver
    def forward(self, z, integration_times):
        z_t = odeint(self.odefunc, z, integration_times, method=self.solver, rtol=1e-4, atol=1e-4)
        return z_t[-1]

# ==============================================================================
# 3. Data Processor (Coordinate Standardization)
# ==============================================================================
class CoordinateProcessor:
    def __init__(self):
        self.stats = {}
        self.train_times = []
    
    def fit(self, coords_list, times):
        self.train_times = sorted([float(t) for t in times])
        for c, t in zip(coords_list, times):
            t = float(t)
            self.stats[t] = {
                'center': np.mean(c, axis=0),
                'scale': np.std(c, axis=0).mean() + 1e-8
            }
        return self

    def transform(self, coords, t):
        t = float(t)
        stats = self._get_interpolated_stats(t)
        return (coords - stats['center']) / stats['scale']

    def inverse_transform(self, rel_coords, t):
        t = float(t)
        stats = self._get_interpolated_stats(t)
        return rel_coords * stats['scale'] + stats['center']
    
    def _get_interpolated_stats(self, t):
        times = np.array(self.train_times)
        if t in self.stats: return self.stats[t]
        if t <= times[0]: return self.stats[times[0]]
        if t >= times[-1]: return self.stats[times[-1]]
        idx = np.searchsorted(times, t)
        t1, t2 = times[idx-1], times[idx]
        w = (t - t1) / (t2 - t1)
        stats1, stats2 = self.stats[t1], self.stats[t2]
        return {
            'center': (1-w)*stats1['center'] + w*stats2['center'],
            'scale': (1-w)*stats1['scale'] + w*stats2['scale']
        }
    
    def to_dict(self):
        return {'stats': self.stats, 'train_times': self.train_times}

    @classmethod
    def from_dict(cls, d):
        processor = cls()
        processor.stats = d['stats']
        processor.train_times = d['train_times']
        return processor

# ==============================================================================
# 4. Main Generator Class
# ==============================================================================
class SpatialMorphogenesisGenerator:
    def __init__(self, device='cpu', time_series_data=None, hidden_dim=256):
        self.device = device
        self.hidden_dim = hidden_dim
        
        if time_series_data is not None:
            self.time_series_data = time_series_data
            self._preprocess_data()
            self._compute_ot_pairs()
            latent_dim = self.all_expr_tensor[0].shape[1]
        else:
            self.coord_processor = None
            self.train_times = []
            latent_dim = 32
            
        self.ode_net = SpatialODENet(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
        self.ode_func = ContinuousTimeODEfunc(self.ode_net)
        self.cnf = ContinuousTimeCNF(self.ode_func).to(device)

    def _preprocess_data(self):
        print("Preprocessing spatial and expression data...")
        self.all_coords = []
        self.all_expr = []
        self.train_times = []
        
        for adata in self.time_series_data:
            t = adata.obs['time_point'].iloc[0] if 'time' in adata.obs else adata.obs['Time'].iloc[0]
            if hasattr(t, 'item'): t = t.item()
            self.train_times.append(float(t))
            
            
            coords = adata.obsm['spatial_aligned_gt'][:, :2].copy()
           
            self.all_coords.append(coords)
            
            expr = adata.X
            if hasattr(expr, 'toarray'): expr = expr.toarray()
            self.all_expr.append(expr.astype(np.float32))
        
        self.coord_processor = CoordinateProcessor()
        self.coord_processor.fit(self.all_coords, self.train_times)
        
        self.all_coords_rel = []
        self.all_expr_tensor = []
        for i, t in enumerate(self.train_times):
            self.all_coords_rel.append(self.coord_processor.transform(self.all_coords[i], t))
            self.all_expr_tensor.append(torch.tensor(self.all_expr[i], dtype=torch.float32).to(self.device))

    def _compute_ot_pairs(self):
        print("Computing OT pairs...")
        self.ot_pairs = {}
        for i in range(len(self.train_times) - 1):
            coords1 = self.all_coords_rel[i]
            coords2 = self.all_coords_rel[i+1]
            idx1, idx2 = optimal_transport_match(coords1, coords2)
            self.ot_pairs[i] = (idx1, idx2)

    def train_spatial_ode(self, epochs=1000, lr=1e-3, batch_size=512):
        if not hasattr(self, 'ot_pairs'):
            print("No training data available.")
            return

        print("\nTraining Spatial ODE (MSE + Chamfer + Feature-Fused FGW)...")
        optimizer = optim.Adam(self.cnf.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Hyperparameters
        CHAMFER_WEIGHT = 0.5
        FGW_WEIGHT = 0.1      
        FGW_MAX_POINTS = 128  # Limit points for FGW to avoid OOM
        
        for epoch in range(epochs):
            self.cnf.train()
            total_loss = 0
            
            for i in range(len(self.train_times) - 1):
                dt = self.train_times[i+1] - self.train_times[i]
                idx1, idx2 = self.ot_pairs[i]
                
                if len(idx1) > batch_size:
                    perm = np.random.permutation(len(idx1))[:batch_size]
                    b_idx1, b_idx2 = idx1[perm], idx2[perm]
                else:
                    b_idx1, b_idx2 = idx1, idx2
                
                # 1. Prepare source state (coords + features)
                src_coords = torch.tensor(self.all_coords_rel[i][b_idx1], dtype=torch.float32).to(self.device)
                src_expr = self.all_expr_tensor[i][b_idx1]
                src_state = torch.cat([src_coords, src_expr], dim=1)
                
                # 2. Prepare target coordinates
                tgt_coords = torch.tensor(self.all_coords_rel[i+1][b_idx2], dtype=torch.float32).to(self.device)
                
                # 3. ODE integration prediction
                pred_full = self.cnf(src_state, torch.tensor([0.0, dt], device=self.device))
                pred_coords = pred_full[:, :2] # Only extract predicted coords
                
                # 4. MSE Loss
                mse_loss = F.mse_loss(pred_coords, tgt_coords)
                
                # 5. Chamfer Loss
                full_tgt_coords = torch.tensor(self.all_coords_rel[i+1], dtype=torch.float32).to(self.device)
                if len(full_tgt_coords) > 1000: 
                    full_tgt_coords = full_tgt_coords[torch.randperm(len(full_tgt_coords))[:1000]]
                dists = torch.cdist(pred_coords, full_tgt_coords)
                chamfer_loss = dists.min(dim=1)[0].mean()
                
                loss = mse_loss + CHAMFER_WEIGHT * chamfer_loss
                
                # 6. FGW Loss (Custom PyTorch implementation)
                if pred_coords.shape[0] > 10:
                    if pred_coords.shape[0] > FGW_MAX_POINTS:
                        idx_fgw = torch.randperm(pred_coords.shape[0])[:FGW_MAX_POINTS]
                        pred_fgw = pred_coords[idx_fgw]
                        src_expr_fgw = src_expr[idx_fgw]
                        
                        idx_tgt_fgw = torch.randperm(full_tgt_coords.shape[0])[:FGW_MAX_POINTS]
                        tgt_fgw = full_tgt_coords[idx_tgt_fgw]
                        tgt_expr_fgw = self.all_expr_tensor[i+1][idx_tgt_fgw]
                    else:
                        pred_fgw = pred_coords
                        src_expr_fgw = src_expr
                        tgt_fgw = full_tgt_coords[:pred_coords.shape[0]]
                        tgt_expr_fgw = self.all_expr_tensor[i+1][:pred_coords.shape[0]]
                    
                    C1 = torch.cdist(pred_fgw, pred_fgw, p=2) + 1e-8
                    C2 = torch.cdist(tgt_fgw, tgt_fgw, p=2) + 1e-8
                    M = torch.cdist(src_expr_fgw, tgt_expr_fgw, p=2)**2
                    
                    h2 = torch.ones(C1.shape[0], device=self.device) / C1.shape[0]
                    h3 = torch.ones(C2.shape[0], device=self.device) / C2.shape[0]
                    
                    # Use custom FGW
                    fgw_loss = compute_fgw_loss(M, C1, C2, h2, h3, alpha=0.5)
                    loss = loss + FGW_WEIGHT * fgw_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cnf.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Total Loss = {total_loss:.6f}")

    def predict(self, start_time_idx, target_time, min_dist=0.01):
        start_time = self.train_times[start_time_idx]
        curr_coords = self.all_coords[start_time_idx]
        curr_expr = self.all_expr_tensor[start_time_idx]
        
        rel_coords = self.coord_processor.transform(curr_coords, start_time)
        rel_tensor = torch.tensor(rel_coords, dtype=torch.float32).to(self.device)
        
        state_tensor = torch.cat([rel_tensor, curr_expr], dim=1)
        dt = target_time - start_time
        
        with torch.no_grad():
            self.cnf.eval()
            pred_full = self.cnf(state_tensor, torch.tensor([0.0, dt], device=self.device))
            pred_rel = to_numpy(pred_full[:, :2]) 
            
        pred_rel = disperse_points(pred_rel, min_dist=min_dist)
        pred_coords = self.coord_processor.inverse_transform(pred_rel, target_time)
        
        adata_pred = ad.AnnData(
            X=np.zeros((len(pred_coords), 1)), 
            obsm={'spatial': pred_coords},
            obs={
                'start_time': [start_time] * len(pred_coords),
                'predicted_time': [target_time] * len(pred_coords)
            }
        )
        return adata_pred

    def save(self, save_path):
        save_dict = {
            'ode_net': self.ode_net.state_dict(),
            'coord_processor': self.coord_processor.to_dict(),
            'train_times': self.train_times,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.all_expr_tensor[0].shape[1]
        }
        torch.save(save_dict, save_path)
        print(f"Model saved to: {save_path}")
    
    @classmethod
    def load(cls, load_path, time_series_data=None, device='cpu'):
        checkpoint = torch.load(load_path, map_location=device)
        hidden_dim = checkpoint.get('hidden_dim', 256)
        generator = cls(device=device, time_series_data=time_series_data, hidden_dim=hidden_dim)
        generator.train_times = checkpoint['train_times']
        generator.coord_processor = CoordinateProcessor.from_dict(checkpoint['coord_processor'])
        generator.ode_net.load_state_dict(checkpoint['ode_net'])
        generator.ode_net.eval()
        print(f"Model loaded from: {load_path}")
        return generator

# ==============================================================================
# 5. Main Program Entry Point
# ==============================================================================
def main():
    BASE_DIR = './sin_dataset/'
    DATA_PATH = os.path.join(BASE_DIR, 'sim_processed_adata_with_scaler.h5ad')
    SAVE_DIR = os.path.join(BASE_DIR)
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'sim_cnf_spatial.pth')
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")
    
    adata = sc.read_h5ad(DATA_PATH)
    time_points = sorted(adata.obs['time_point'].unique())
    time_series_data = [adata[adata.obs['time_point'] == tp].copy() for tp in time_points]
    
    print("\n" + "=" * 50)
    print("Initializing and training model...")
    generator = SpatialMorphogenesisGenerator(time_series_data=time_series_data, device=DEVICE)
    generator.train_spatial_ode(epochs=1000, lr=1e-3)
    
    print("\n" + "=" * 50)
    generator.save(MODEL_SAVE_PATH)
    print("Done!")

if __name__ == "__main__":
    main()
