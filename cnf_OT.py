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
import pandas as pd

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 基础工具函数
# ==============================================================================

def to_numpy(x):
    if x is None: return None
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray): return x
    return np.array(x)

def disperse_points(points, min_dist=0.01, max_iter=100):
    """物理排斥算法"""
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
    """最优传输匹配"""
    n1, n2 = len(coords1), len(coords2)
    if n1 > max_points: idx1 = np.random.choice(n1, max_points, replace=False)
    else: idx1 = np.arange(n1)
    if n2 > max_points: idx2 = np.random.choice(n2, max_points, replace=False)
    else: idx2 = np.arange(n2)
        
    cost = cdist(coords1[idx1], coords2[idx2])
    row_ind, col_ind = linear_sum_assignment(cost)
    return idx1[row_ind], idx2[col_ind]

# ==============================================================================
# 1. 网络模型定义
# ==============================================================================


class GrowthNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(129, 64) 
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# 1. 必须复制模型类定义 (与训练代码完全一致)
# ==========================================
# 如果没有这些类定义，load_state_dict 无法知道要把权重加载到什么结构里

class SinusoidaltimeEmbeddings(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim

    def forward(self, time):
        device = time.device
        half_dim = self.time_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class BalancedClassifier(nn.Module):
    def __init__(self, gene_dim, spatial_dim, time_emb_dim=64, hidden_dims=[512, 256], output_dim=27):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidaltimeEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.Mish(),
        )
        input_dim = gene_dim + spatial_dim
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Mish(),
            nn.Dropout(0.25)
        )
        combined_dim = hidden_dims[0] + time_emb_dim
        self.layer2 = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Mish(),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Linear(hidden_dims[1], output_dim)
        
    def forward(self, x, z, t):
        t_emb = self.time_mlp(t)
        features = torch.cat([x, z], dim=1)
        out = self.layer1(features)
        out = torch.cat([out, t_emb], dim=1)
        out = self.layer2(out)
        logits = self.classifier(out)
        return logits

# ==========================================
# 2. 修改后的预测函数
# ==========================================

def model_anno(model_path, x, t):
    """
    加载模型并预测新样本
    参数:
        model_path: .pth 文件路径
        x: numpy array, 形状 (N, Gene_Dim + 2)。假设最后两列是空间坐标。
        t: numpy array, 形状 (N, ) 或 (N, 1)。对应的时间点。
    返回:
        logits: 模型的原始输出
    """
    
    # 1. 加载 Checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # 2. 提取预处理器
    if "preprocessors" not in checkpoint:
        raise KeyError("模型文件中找不到 'preprocessors'，请确认使用了最新的训练代码。")
        
    preprocessors = checkpoint["preprocessors"]
    # 注意：最新训练代码使用的是简写键名
    gene_scaler = preprocessors["gs"]     # 对应 gene_scaler
    spatial_scaler = preprocessors["ss"]  # 对应 spatial_scaler
    time_scaler = preprocessors["ts"]     # 对应 time_scaler
    label_encoder = preprocessors["le"]   # 对应 label_encoder
    
    # 3. 动态推断模型维度 (从 scaler 中获取)
    # gene_scaler.mean_ 的长度就是训练时的基因维度
    gene_dim = gene_scaler.mean_.shape[0]
    spatial_dim = spatial_scaler.mean_.shape[0]
    num_classes = len(label_encoder.classes_)
    
    # 4. 重新构建模型并加载权重
    model = BalancedClassifier(
        gene_dim=gene_dim,
        spatial_dim=spatial_dim,
        output_dim=num_classes
    )
    # 加载权重字典
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # 切换到评估模式
    
    # 5. 数据拆分 (假设 x 包含了 基因 + 空间)
    # x 的最后两列是空间坐标，前面是基因
    x_genes = x[:, :-2]
    x_spatial = x[:, -2:]
    
    # 确保时间 t 的形状是 (N, 1)
    if t.ndim == 1:
        t = t.reshape(-1, 1)
        
    # 6. 预处理 (Transform)
    # 注意：必须转为 numpy 才能进 sklearn scaler
    x_genes_scaled = gene_scaler.transform(x_genes)
    x_spatial_scaled = spatial_scaler.transform(x_spatial)
    t_scaled = time_scaler.transform(t)
    
    # 7. 转 Tensor 并预测
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x_genes_scaled)
        z_tensor = torch.FloatTensor(x_spatial_scaled)
        t_tensor = torch.FloatTensor(t_scaled)
        
        logits = model(x_tensor, z_tensor, t_tensor)
        
        # 如果需要概率或类别，可以在这里处理
        # probs = torch.softmax(logits, dim=1)
        # preds = torch.argmax(probs, dim=1)
        # pred_labels = label_encoder.inverse_transform(preds.numpy())
        
    return logits

def get_cell_type(cur_cell, model_anno_path, t, cell_type_unique):
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 确保 cur_cell 是 Tensor
    if not isinstance(cur_cell, torch.Tensor):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cur_cell = torch.tensor(cur_cell, dtype=torch.float32, device=device)
    else:
        device = cur_cell.device

    with torch.no_grad():
        # 构造时间张量，确保在同一设备
        t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
        if t_tensor.ndim == 0:
            time = t_tensor.view(1, 1).repeat(cur_cell.shape[0], 1)
        else:
            time = t_tensor
        
        # 预测 Logits
        outputs = model_anno(model_anno_path, cur_cell, time)
        
        # 获取索引
        cell_type_index = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        
        # [关键修复] 将索引移回 CPU 转为 numpy，才能用来索引 list/array
        cell_type_index_cpu = cell_type_index.cpu().numpy()
        
        if cell_type_unique is not None:
            cur_cell_type = list(cell_type_unique[cell_type_index_cpu])
        else:
            cur_cell_type = cell_type_index_cpu # 如果没有标签名，返回索引
            
    return cur_cell_type

class SpatialODENet(nn.Module):
    """空间流场网络"""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), # x, y, t
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        nn.init.normal_(self.net[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.net[-1].bias, 0)
    
    def forward(self, t, y):
        batch_size = y.shape[0]
        t_expanded = t * torch.ones(batch_size, 1).to(y.device)
        x_with_time = torch.cat([y, t_expanded], dim=1)
        return self.net(x_with_time)

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
# 2. 数据处理器
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
        
        stats1 = self.stats[t1]
        stats2 = self.stats[t2]
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
# 3. 核心业务逻辑
# ==============================================================================

def resample_cells(coords, expr, growth_model, t, delta_t=0.2, sigma_d=0.00001):
    cell_number = expr.shape[0]
    x_growth = []
    y_growth = []
    new_cell_id = []
    cur_cell_id = np.arange(expr.shape[0])
    
    device = next(growth_model.parameters()).device
    
    # 维度匹配
    expr_dim_needed = 128 
    if expr.shape[1] > expr_dim_needed:
        expr_input = expr[:, :expr_dim_needed]
    else:
        expr_input = expr
        
    expr_tensor_input = torch.tensor(expr_input, dtype=torch.float32).to(device)
    t_tensor = torch.tensor(t, dtype=torch.float32).view(1, 1).to(device).repeat(cell_number, 1)
    combined_input = torch.cat([expr_tensor_input, t_tensor], axis=1)
    
    with torch.no_grad():
        g = growth_model(combined_input)
    
    g = g.cpu().numpy().flatten()
    expr_np = expr 
    spatial_np = coords
    
    for i in range(cell_number):
        g_i = g[i]
        temp = np.random.rand()
        
        if g_i > 1.0 and temp < (g_i * delta_t):
            # 保留母细胞
            x_growth.append(expr_np[i])
            y_growth.append(spatial_np[i])
            new_cell_id.append(cur_cell_id[i])
            # 子细胞
            new_expr = expr_np[i] + sigma_d * np.random.randn(*expr_np[i].shape)
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.exponential(0.1) + 0.1
            offset = np.array([np.cos(angle), np.sin(angle)]) * distance
            new_spatial = spatial_np[i] + offset + sigma_d * np.random.randn(2)
            
            x_growth.append(new_expr)
            y_growth.append(new_spatial)
            new_cell_id.append(f"{cur_cell_id[i]}_child")
        elif not (g_i < 1.0 and temp < (-g_i * delta_t)):
            # 存活
            x_growth.append(expr_np[i])
            y_growth.append(spatial_np[i])
            new_cell_id.append(cur_cell_id[i])
            
    if len(x_growth) == 0: return coords, expr, cur_cell_id
    return np.array(y_growth), np.array(x_growth), new_cell_id


# ==============================================================================
# 4. 主生成器类
# ==============================================================================

class MorphogenesisGenerator:
    def __init__(self, growth_model, expr_model, model_anno_path, cell_type_unique, device='cpu', 
                 time_series_data=None, hidden_dim=256):
        """
        初始化
        """
        self.device = device
        self.hidden_dim = hidden_dim
        self.growth_model = growth_model.to(device).eval()
        self.expr_model = expr_model
        self.model_anno_path = model_anno_path
        self.cell_type_unique = cell_type_unique # 保存细胞类型列表
        
        #print(f"Loading Annotation Model from {model_anno_path}")
        #loaded = torch.load(model_anno_path, map_location=device)
        #self.anno_model = loaded["complete_model"].to(device).eval()
        
        # 初始化网络
        self.ode_net = SpatialODENet(hidden_dim=hidden_dim).to(device)
        self.ode_func = ContinuousTimeODEfunc(self.ode_net)
        self.cnf = ContinuousTimeCNF(self.ode_func).to(device)
        
        # 如果提供了数据，则初始化处理器和训练数据
        if time_series_data is not None:
            self.time_series_data = time_series_data
            self._preprocess_data()
            self._compute_ot_pairs()
        else:
            self.coord_processor = None
            self.train_times = []

    def _preprocess_data(self):
        print("Preprocessing spatial data...")
        self.all_coords = []
        self.all_expr = []
        self.train_times = []
        
        for adata in self.time_series_data:
            t = adata.obs['time'].iloc[0] if 'time' in adata.obs else adata.obs['time'].iloc[0]
            if hasattr(t, 'item'): t = t.item()
            self.train_times.append(float(t))
            
            if 'X_spatial_aligned' in adata.obsm:
                coords = adata.obsm['X_spatial_aligned'][:, :2].copy()
            else:
                coords = adata.obsm['spatial'][:, :2].copy()
            self.all_coords.append(coords)
            
            expr = adata.obsm['X_latent'] 
            if hasattr(expr, 'toarray'): expr = expr.toarray()
            self.all_expr.append(expr)
        
        self.coord_processor = CoordinateProcessor()
        self.coord_processor.fit(self.all_coords, self.train_times)
        
        self.all_coords_rel = []
        for i, t in enumerate(self.train_times):
            self.all_coords_rel.append(self.coord_processor.transform(self.all_coords[i], t))

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

        print("\nTraining Spatial ODE...")
        optimizer = optim.Adam(self.cnf.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
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
                
                src = torch.tensor(self.all_coords_rel[i][b_idx1], dtype=torch.float32).to(self.device)
                tgt = torch.tensor(self.all_coords_rel[i+1][b_idx2], dtype=torch.float32).to(self.device)
                
                pred = self.cnf(src, torch.tensor([0.0, dt], device=self.device))
                
                mse_loss = F.mse_loss(pred, tgt)
                
                # Chamfer Loss
                full_tgt = torch.tensor(self.all_coords_rel[i+1], dtype=torch.float32).to(self.device)
                if len(full_tgt) > 1000: full_tgt = full_tgt[torch.randperm(len(full_tgt))[:1000]]
                dists = torch.cdist(pred, full_tgt)
                chamfer_loss = dists.min(dim=1)[0].mean()
                
                loss = mse_loss + 0.5 * chamfer_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cnf.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.6f}")

    def predict(self, start_time_idx, target_time, min_dist=0.01):
        start_time = self.train_times[start_time_idx]
        
        curr_coords = self.all_coords[start_time_idx]
        curr_expr = self.all_expr[start_time_idx]
        
        # 1. Growth
        res_coords, res_expr, new_ids = resample_cells(
            curr_coords, curr_expr, self.growth_model, start_time
        )
        
        # 2. Gene Evolution
        integration_times = torch.tensor([start_time, target_time], dtype=torch.float32, device=self.device)
        res_expr_tensor = torch.from_numpy(res_expr).float().to(self.device)
        zero_pad = torch.zeros(res_expr_tensor.shape[0], 1, device=self.device)
        
        with torch.no_grad():
            pred_expr, _ = self.expr_model.chain[0](
                res_expr_tensor, zero_pad, integration_times=integration_times, reverse=False
            )
            pred_expr_np = to_numpy(pred_expr)

        # 3. Spatial ODE
        rel_coords = self.coord_processor.transform(res_coords, start_time)
        rel_tensor = torch.tensor(rel_coords, dtype=torch.float32, device=self.device)
        
        dt = target_time - start_time
        with torch.no_grad():
            self.cnf.eval()
            pred_rel_tensor = self.cnf(rel_tensor, torch.tensor([0.0, dt], device=self.device))
            pred_rel = to_numpy(pred_rel_tensor)
            
        # 4. Physics
        pred_rel = disperse_points(pred_rel, min_dist=min_dist)
        
        # 5. Inverse Transform
        pred_coords = self.coord_processor.inverse_transform(pred_rel, target_time)
        expr_tensor = torch.tensor(pred_expr_np, dtype=torch.float32, device=self.device)
        coords_tensor = torch.tensor(pred_coords, dtype=torch.float32, device=self.device)

        # 6. Annotation
        combined_input = torch.cat([expr_tensor, coords_tensor], axis=1)
        annotation = get_cell_type(combined_input, self.model_anno_path, target_time, self.cell_type_unique)
        
        adata_pred = ad.AnnData(
            X=pred_expr_np,
            obsm={
                'spatial': pred_coords,
                'X_latent': pred_expr_np
                
            },
            obs={
                'Annotation': annotation
            },
        )
       

        return adata_pred

    def visualize(self, start_time_idx, target_times, save_path=None):
        """可视化并返回预测结果"""
        predictions = []
        for t in target_times:
            predictions.append(self.predict(start_time_idx, t))
        
        n_plots = len(target_times) + 1
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        
        # 起始点
        coords = self.all_coords[start_time_idx]
        axes[0].scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.6, c='blue')
        axes[0].set_title(f'Start (t={self.train_times[start_time_idx]})')
        axes[0].set_aspect('equal')
        
        # 预测点
        for i, (t, pred) in enumerate(zip(target_times, predictions)):
            c = pred.obsm['spatial']
            axes[i + 1].scatter(c[:, 0], c[:, 1], s=5, alpha=0.6, c='red')
            axes[i + 1].set_title(f'Pred (t={t:.2f})')
            axes[i + 1].set_aspect('equal')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存: {save_path}")
        plt.show()
        
        return predictions

    # ==================== 保存和加载 ====================
    
    def save(self, save_path):
        """
        保存模型
        """
        save_dict = {
            # 网络权重 (注意：这里保存的是 ODE Net)
            'ode_net': self.ode_net.state_dict(),
            
            # 坐标处理器
            'coord_processor': self.coord_processor.to_dict(),
            
            # 配置
            'train_times': self.train_times,
            'hidden_dim': self.hidden_dim,
        }
        
        torch.save(save_dict, save_path)
        print(f"模型已保存: {save_path}")
    
    @classmethod
    def load(cls, load_path, growth_model, expr_model, model_anno_path, 
             time_series_data=None, cell_type_unique=None, device='cpu'):
        """
        加载模型
        """
        checkpoint = torch.load(load_path, map_location=device)
        
        # 从 checkpoint 中读取 hidden_dim
        hidden_dim = checkpoint.get('hidden_dim', 256)
        
        # 1. 创建生成器实例
        generator = cls(
            growth_model=growth_model,
            expr_model=expr_model,
            model_anno_path=model_anno_path,
            device=device,
            time_series_data=time_series_data,
            cell_type_unique=cell_type_unique, # <--- 修复: 传入 cell_type_unique
            hidden_dim=hidden_dim
        )
        
        # 2. 覆盖/恢复关键属性
        generator.train_times = checkpoint['train_times']
        
        # 恢复坐标处理器
        generator.coord_processor = CoordinateProcessor.from_dict(checkpoint['coord_processor'])
        
        # 恢复网络权重
        generator.ode_net.load_state_dict(checkpoint['ode_net'])
        generator.ode_net.eval()
        
        print(f"模型已加载: {load_path}")
        print(f"训练时间点: {generator.train_times}")
        
        return generator

# ==============================================================================
# 5. 主程序
# ==============================================================================

def main():
    # 配置
    BASE_DIR = '/home/lenovo/jora/data'
    DATA_PATH = os.path.join(BASE_DIR, 'R5_filtered_latent.h5ad')
    GROWTH_MODEL_PATH = os.path.join(BASE_DIR, 'grow_model_2.pth')
    EXPR_MODEL_PATH = os.path.join(BASE_DIR, 'Rgraph_checkpt.pth')
    ANNO_MODEL_PATH = os.path.join(BASE_DIR, 'model_anno.pth')
    
    SAVE_DIR = os.path.join(BASE_DIR, 'results_final_struct')
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'shape_generator_ode_pca.pth')
    os.makedirs(SAVE_DIR, exist_ok=True)

    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")
    
    # 1. 加载数据
    adata = sc.read_h5ad(DATA_PATH)
    time_points = sorted(adata.obs['time'].unique())
    time_series_data = [adata[adata.obs['time'] == tp].copy() for tp in time_points]
    
    # 获取细胞类型列表 (用于 Annotation)
    if 'Annotation' in adata.obs:
        cell_type_unique = adata.obs['Annotation'].cat.categories.values
        print(f"Unique Cell Types: {len(cell_type_unique)}")
    else:
        cell_type_unique = None
        print("Warning: Annotation column not found.")

    # 2. 加载外部模型
    growth_model = torch.load(GROWTH_MODEL_PATH, map_location=DEVICE)
    expr_model = torch.load(EXPR_MODEL_PATH, map_location=DEVICE)
    
    # 3. 初始化并训练
    print("\n" + "=" * 50)
    print("初始化并训练模型...")
    generator = MorphogenesisGenerator(
        growth_model=growth_model,
        expr_model=expr_model,
        model_anno_path=ANNO_MODEL_PATH,
        time_series_data=time_series_data,
        cell_type_unique=cell_type_unique,
        device=DEVICE
    )
    
    generator.train_spatial_ode(epochs=1000, lr=1e-3)
    
    # 4. 保存模型
    print("\n" + "=" * 50)
    generator.save(MODEL_SAVE_PATH)
    
    # 5. 演示：加载模型 (已修复 NoneType 错误)
    print("\n" + "=" * 50)
    print("演示从文件加载模型...")
    
    loaded_generator = MorphogenesisGenerator.load(
        load_path=MODEL_SAVE_PATH,
        growth_model=growth_model,
        expr_model=expr_model,
        model_anno_path=ANNO_MODEL_PATH,
        time_series_data=time_series_data, 
        cell_type_unique=cell_type_unique, # <--- 必须传入，否则推理时无法获取标签名
        device=DEVICE
    )
    
    # 6. 预测并可视化
    print("\n" + "=" * 50)
    print("预测并可视化...")
    
    # 假设从第 2 个时间点 (index 1) 开始预测
    start_time_idx = 1 
    
    target_times = [1.2,1.4,1.6, 1.8, 2.0]
    
    predictions = loaded_generator.visualize(
        start_time_idx=start_time_idx,
        target_times=target_times,
        save_path=os.path.join(SAVE_DIR, 'predictions_ode1_2.png')
    )
    
    # 7. 保存预测结果
    print("\n" + "=" * 50)
    print("保存预测结果...")
    
    for pred, t in zip(predictions, target_times):
        path = os.path.join(SAVE_DIR, f'predicted_t{t:.1f}.h5ad')
        pred.write(path)
        print(f"  已保存: {path}")

    print("\n" + "=" * 50)
    print("全部完成!")

if __name__ == "__main__":
    main()