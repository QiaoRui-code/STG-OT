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
   
# GAE模型代码
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
        """获取潜在表示"""
        with torch.no_grad():
            _, z = self.forward(x, adj, idx)
        return z
    
    def forward_loss(self, x, adj, idx):
        y, z = self.forward(x, adj, idx)
        x = x[idx, :]
        loss = F.mse_loss(y, x)
        return loss, y, z

def create_joint_representation_and_update_adata(adata, dim_latent=64, model_save_path=None, n_epochs=100):
    """
    创建联合表示并更新AnnData对象
    
    参数:
        adata: AnnData对象，包含原始基因表达和空间坐标
        dim_latent: 潜在表示的维度
        model_save_path: 模型保存路径，如果为None则不保存
        n_epochs: 训练轮数
    
    返回:
        updated_adata: 更新后的AnnData对象，其中X被替换为联合表示，obsm['spatial']被替换为归一化坐标
    """
    # 设置设备
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 1. 从adata中提取数据
    expression_data = adata.X
    if sp.issparse(expression_data):
        expression_data = expression_data.toarray()
    
    spatial_coords = adata.obsm['spatial']
    
    # 2. 数据预处理
    # 归一化基因表达数据
    #expr_scaler = StandardScaler()
    expr_normalized = expression_data
    
    # 归一化空间坐标
    coord_scaler = StandardScaler()
    coords_normalized = coord_scaler.fit_transform(spatial_coords)
    
    # 3. 构建邻接矩阵 (基于空间坐标)
    adj_matrix = kneighbors_graph(
        coords_normalized, 
        n_neighbors=5, 
        mode='connectivity', 
        include_self=True
    )
    adj_matrix = adj_matrix.toarray()
    
    # 4. 转换为PyTorch张量
    X = torch.FloatTensor(expr_normalized).to(device)
    adj = torch.FloatTensor(adj_matrix).to(device)
    idx = torch.arange(X.shape[0]).to(device)  # 所有细胞的索引
    
    # 5. 初始化GAE模型
    n_genes = expression_data.shape[1]
    model = GAE(
        dim_in=n_genes,
        dim_hidden=[1024, 512],
        dim_latent=dim_latent,
        in_logarithm=True
    ).to(device)
    
    # 6. 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("开始训练GAE模型...")
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss, _, _ = model.forward_loss(X, adj, idx)
        loss.backward()
        optimizer.step()
        
        # 清理GPU内存
        torch.cuda.empty_cache() 
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # 垃圾回收
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
    
    # 保存模型（如果指定了路径）
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'dim_in': n_genes,
            'dim_hidden': [1024, 512],
            'dim_latent': dim_latent,
            'in_logarithm': True
        }, model_save_path)
        print(f"模型已保存到: {model_save_path}")
    
    # 7. 获取潜在表示
    print("获取潜在表示...")
    model.eval()
    with torch.no_grad():
        latent_representations = model.get_latent_representation(X, adj, idx)
    latent_representations = latent_representations.cpu().numpy()
    
    # 8. 创建联合表示：将潜在变量与归一化坐标连接
    print("创建联合表示...")
    joint_representation = np.concatenate([latent_representations, coords_normalized], axis=1)
    
    # 9. 创建新的AnnData对象，保留所有原始信息但替换X和obsm['spatial']
    print("创建更新后的AnnData对象...")
    
    # 创建新的var（特征）信息
    new_var_names = [f'latent_{i}' for i in range(dim_latent)] + ['coord_x', 'coord_y']
    new_var = pd.DataFrame(index=new_var_names)
    new_var['feature_type'] = ['latent'] * dim_latent + ['coord'] * 2
    
    # 创建新的AnnData对象，复制所有原始信息
    updated_adata = ad.AnnData(
        X=joint_representation,
        obs=adata.obs.copy(),           # 复制样本注释
        var=new_var,                    # 使用新的特征信息
        uns=adata.uns.copy(),           # 复制非结构化注释
        obsm=adata.obsm.copy(),         # 复制样本的多维注释
        obsp=adata.obsp.copy(),         # 复制样本间的关系
        layers=adata.layers.copy(),     # 复制其他数据层
        varm=adata.varm.copy(),         # 复制特征的多维注释
        varp=adata.varp.copy()          # 复制特征间的关系
    )
    
    # 更新obsm中的空间坐标为归一化坐标，同时保留原始坐标
    updated_adata.obsm['spatial'] = coords_normalized
    updated_adata.obsm['spatial_original'] = spatial_coords  # 保留原始坐标
    
    # 添加额外的信息到uns中
    updated_adata.uns['gae_info'] = {
        'dim_latent': dim_latent,
        'coord_scaler_mean': coord_scaler.mean_,
        'coord_scaler_scale': coord_scaler.scale_,
        'model_save_path': model_save_path,
        'n_epochs': n_epochs
    }
    
    # 添加潜在表示和归一化坐标到obsm中，以便后续使用
    updated_adata.obsm['X_latent'] = latent_representations
    updated_adata.obsm['coords_normalized'] = coords_normalized
    
    # 确保原始adata中的所有信息都被保留
    # 检查并复制可能遗漏的属性
    if hasattr(adata, 'raw'):
        updated_adata.raw = adata.raw
    
    return updated_adata

# 示例使用
if __name__ == "__main__":
    # 加载您的数据
    adata = sc.read("/home/lenovo/jora/GAE/RTime_aligned_rotated.h5ad")
    model_save_path = "/home/lenovo/jora/GAE/ae_model2.pth"
    
    # 创建联合表示并更新AnnData对象
    updated_adata = create_joint_representation_and_update_adata(
        adata, 
        dim_latent=32, 
        model_save_path=model_save_path,
        n_epochs=100
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
    output_path = "/home/lenovo/jora/GAE/RTime_aligned_rotated_latent.h5ad"
    updated_adata.write(output_path)
    print(f"更新后的数据已保存到: {output_path}")