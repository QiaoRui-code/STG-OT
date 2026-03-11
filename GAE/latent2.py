import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ------------------------------
# 原模型结构不变（GAE核心逻辑无需修改）
# ------------------------------
class content_graph_conv(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(content_graph_conv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(in_features, out_features))))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(out_features,))))
        else:
            self.register_parameter('bias', None)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias
        return torch.mm(adj, x)

class Encoder(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: list, dim_latent: int, act_fn: object = nn.LeakyReLU):
        super().__init__()
        self.net = self._build_layers(dim_in, dim_hidden, dim_latent)
    
    def _build_layers(self, dim_in, dim_hidden, dim_out):
        if len(dim_hidden) == 0:
            return nn.Sequential(nn.Linear(dim_in, dim_out))
        return nn.Sequential(
            nn.Linear(dim_in, dim_hidden[0]),
            nn.ReLU(),
            self._build_layers(dim_hidden[0], dim_hidden[1:], dim_out)
        )
    
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: list, dim_out: int, act_fn: object = nn.ReLU):
        super().__init__()
        self.act_fn = act_fn()
        self.net = self._build_layers(dim_in, dim_hidden, dim_out)

    def _build_layers(self, dim_in, dim_hidden, dim_out):
        if len(dim_hidden) == 0:
            return nn.Sequential(nn.Linear(dim_in, dim_out))
        return nn.Sequential(
            nn.Linear(dim_in, dim_hidden[0]),
            nn.ReLU(),
            self._build_layers(dim_hidden[0], dim_hidden[1:], dim_out)
        )
    
    def forward(self, x):
        x = self.net(x)
        return self.act_fn(x)

class GAE(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_latent, in_logarithm=True, act_fn=nn.LeakyReLU):
        super(GAE, self).__init__()
        self.gcn_layer = content_graph_conv(dim_in, dim_hidden[0])
        self.encoder = Encoder(dim_hidden[0], dim_hidden[1:], dim_latent, act_fn)
        self.decoder = Decoder(dim_latent, dim_hidden[::-1], dim_in, act_fn)
        self.in_logarithm = in_logarithm
        self.BN = nn.BatchNorm1d(dim_hidden[0])

    def forward(self, x, adj, idx):
        if self.in_logarithm:
            x = torch.log(1 + x) 
        h = F.relu(self.BN(self.gcn_layer(x, adj)))
        h = h[idx, :]
        z = self.encoder(h)
        out = self.decoder(z)
        return out, z

    def generate(self, x, adj, idx):
        return self.forward(x, adj, idx)
    
    def get_latent_representation(self, x, adj, idx):
        with torch.no_grad():
            _, z = self.forward(x, adj, idx)
        return z
    
    def forward_loss(self, x, adj, idx):
        y, z = self.forward(x, adj, idx)
        x = x[idx, :]
        loss = F.mse_loss(y, x)
        return loss, y, z

# ------------------------------
# 新增：加载模型函数（用于后续解码）
# ------------------------------
def load_gae_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = GAE(
        dim_in=checkpoint['dim_in'],
        dim_hidden=checkpoint['dim_hidden'],
        dim_latent=checkpoint['dim_latent'],
        in_logarithm=checkpoint['in_logarithm']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# ------------------------------
# 核心修改：适配无空间坐标的单细胞数据
# ------------------------------
def create_joint_representation_sc(adata, dim_latent=64, model_save_path=None, n_epochs=100, k_neighbors=10):
    """
    适配无空间坐标的单细胞数据：基于表达相似性构建邻接矩阵，学习潜在表示
    
    参数:
        adata: AnnData对象，仅包含基因表达数据（无需spatial）
        dim_latent: 潜在表示的维度
        model_save_path: 模型保存路径，None则不保存
        n_epochs: 训练轮数
        k_neighbors: kNN邻接矩阵的近邻数（关键参数，可调整）
    
    返回:
        updated_adata: 更新后的AnnData对象，X=联合表示（潜在表示），obsm['X_latent']=潜在表示
    """
    # 设置设备
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 1. 从adata中提取基因表达数据（无空间坐标）
    expression_data = adata.X
    if sp.issparse(expression_data):
        expression_data = expression_data.toarray()  # 稀疏矩阵转稠密（单细胞数据通常可承受）
    
    # 2. 数据预处理（强化表达数据标准化，因为邻接矩阵依赖表达）
    expr_scaler = StandardScaler()  # 启用标准化（关键！消除基因表达量差异）
    expr_normalized = expr_scaler.fit_transform(expression_data)  # (n_cells, n_genes)
    
    # 3. 基于表达相似性构建邻接矩阵（核心修改：替代空间kNN）
    print(f"基于表达相似性构建kNN邻接矩阵（k={k_neighbors}）...")
    adj_matrix = kneighbors_graph(
        expr_normalized,  # 输入：标准化后的表达数据
        n_neighbors=k_neighbors,
        mode='connectivity',
        include_self=True,  # 包含自身（每个细胞是自己的邻居）
        metric='cosine'  # 距离度量：余弦相似度（适合高维表达数据）
    )
    adj_matrix = adj_matrix.toarray()  # 转为稠密矩阵（GCN需要）
    
    # 4. 转换为PyTorch张量
    X = torch.FloatTensor(expr_normalized).to(device)
    adj = torch.FloatTensor(adj_matrix).to(device)
    idx = torch.arange(X.shape[0]).to(device)  # 所有细胞的索引（无筛选，全部参与）
    
    # 5. 初始化模型（结构不变）
    n_genes = expression_data.shape[1]
    model = GAE(
        dim_in=n_genes,
        dim_hidden=[512, 256],  # 可根据数据调整（细胞数多可增大，少则减小）
        dim_latent=dim_latent,
        in_logarithm=True
    ).to(device)
    
    # 6. 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 学习率略调低（表达邻接更敏感）
    print("开始训练GAE模型...")
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss, _, _ = model.forward_loss(X, adj, idx)
        loss.backward()
        optimizer.step()
        
        # 清理内存
        torch.cuda.empty_cache()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # 保存模型
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'dim_in': n_genes,
            'dim_hidden': [1024, 512],  # 与模型初始化一致
            'dim_latent': dim_latent,
            'in_logarithm': True,
            'expr_scaler_mean': expr_scaler.mean_,  # 保存标准化参数（用于后续解码）
            'expr_scaler_scale': expr_scaler.scale_
        }, model_save_path)
        print(f"模型已保存到: {model_save_path}")
    
    # 7. 获取潜在表示
    print("获取潜在表示...")
    model.eval()
    with torch.no_grad():
        latent_representations = model.get_latent_representation(X, adj, idx).cpu().numpy()
    
    # 8. 更新AnnData对象（无空间坐标相关字段）
    updated_adata = ad.AnnData(
        X=latent_representations,  # X替换为潜在表示（联合表示=潜在表示，无坐标拼接）
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=[f'latent_{i}' for i in range(dim_latent)]),  # 新var：潜在特征
        uns=adata.uns.copy()
    )
    
    # 添加关键信息到obsm/uns
    updated_adata.obsm['X_latent'] = latent_representations  # 保留潜在表示
    updated_adata.uns['gae_info'] = {
        'dim_latent': dim_latent,
        'k_neighbors': k_neighbors,
        'n_epochs': n_epochs,
        'model_save_path': model_save_path,
        'expr_scaler_mean': expr_scaler.mean_,
        'expr_scaler_scale': expr_scaler.scale_
    }
    
    # 保留原始adata的raw（如果有）
    if hasattr(adata, 'raw'):
        updated_adata.raw = adata.raw
    
    return updated_adata

# ------------------------------
# 解码函数（适配无空间坐标，从潜在表示恢复基因表达）
# ------------------------------
def decode_latent_to_expression_sc(updated_adata, model_path, device=None):
    """
    从潜在表示解码回原始基因表达数据（无空间坐标版）
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型和标准化参数
    checkpoint = torch.load(model_path, map_location=device)
    model = load_gae_model(model_path, device)
    latent_z = updated_adata.obsm['X_latent']
    expr_scaler_mean = checkpoint['expr_scaler_mean']
    expr_scaler_scale = checkpoint['expr_scaler_scale']
    
    # 解码
    z_tensor = torch.FloatTensor(latent_z).to(device)
    with torch.no_grad():
        decoded_log = model.decoder(z_tensor).cpu().numpy()  # log(1+标准化表达)
    
    # 逆变换：恢复到原始表达尺度
    # 步骤1：exp(decoded_log) - 1 → 标准化后的原始表达
    expr_normalized_recon = np.exp(decoded_log) - 1
    # 步骤2：逆标准化 → 原始表达尺度
    reconstructed_expression = expr_normalized_recon * expr_scaler_scale + expr_scaler_mean
    
    return reconstructed_expression, decoded_log

# ------------------------------
# 示例使用（无空间坐标的单细胞数据）
# ------------------------------
if __name__ == "__main__":
    # 加载您的数据
    adata = sc.read('/home/lenovo/jora/colon_data6.h5ad')
    print(adata)
    model_save_path = "/home/lenovo/jora//ae_model_colon.pth"
    # 稀疏矩阵转为稠密矩阵（仅推荐小数据，或确认内存足够时）
    # 检查稀疏矩阵
    if sp.issparse(adata.X):
    # 转换为数组检查（注意内存）
      X_data = adata.X.toarray()
    else:
      X_data = adata.X

    print(f"数据中NaN的数量: {np.isnan(X_data).sum()}")
    print(f"数据中Inf的数量: {np.isinf(X_data).sum()}")

# 检查数据范围
    print(f"数据最小值: {X_data.min()}")
    print(f"数据最大值: {X_data.max()}")

    # 创建联合表示并更新AnnData对象
    updated_adata =create_joint_representation_sc(
        adata, 
        dim_latent=128, 
        model_save_path=model_save_path,
        n_epochs=1000
    )
    
    # 检查结果
    print(f"原始adata形状: {adata.shape}")
    print(f"更新后的adata形状: {updated_adata.shape}")
    print(f"原始X矩阵前5个特征: {adata.var.index[:5].tolist()}")
    print(f"更新后X矩阵前5个特征: {updated_adata.var.index[:5].tolist()}")
    print(f"更新后X矩阵最后2个特征: {updated_adata.var.index[-2:].tolist()}")
    
    # 检查其他信息是否保留
    print(f"obs信息是否保留: {all(adata.obs.columns == updated_adata.obs.columns)}")
    print(f"uns信息是否保留: {set(adata.uns.keys()).issubset(set(updated_adata.uns.keys()))}")
    print(f"obsm信息是否保留: {set(adata.obsm.keys()).issubset(set(updated_adata.obsm.keys()))}")
    
    # 保存为h5ad文件
    output_path = "/home/lenovo/jora/colon_sc_128.h5ad"
    updated_adata.write(output_path)
    print(f"更新后的数据已保存到: {output_path}")