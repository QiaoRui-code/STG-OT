import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ------------------------------
# VAE模型（可逆降维）
# ------------------------------
class Encoder(nn.Module):
    """VAE编码器：输出均值和对数方差"""
    def __init__(self, dim_in, dim_hidden, dim_latent):
        super().__init__()
        
        # 构建隐藏层
        layers = []
        prev_dim = dim_in
        for h_dim in dim_hidden:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        
        self.hidden = nn.Sequential(*layers)
        
        # 均值和对数方差层
        self.fc_mu = nn.Linear(prev_dim, dim_latent)
        self.fc_logvar = nn.Linear(prev_dim, dim_latent)
    
    def forward(self, x):
        h = self.hidden(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """VAE解码器：从潜在空间重建原始数据"""
    def __init__(self, dim_latent, dim_hidden, dim_out):
        super().__init__()
        
        layers = []
        prev_dim = dim_latent
        for h_dim in reversed(dim_hidden):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, dim_out))
        # 不加激活函数，因为输出是标准化后的表达值（可正可负）
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    """
    变分自编码器（VAE）
    - 编码：原始表达 → 潜在表示
    - 解码：潜在表示 → 重建表达
    - 完全可逆！
    """
    def __init__(self, dim_in, dim_hidden, dim_latent, beta=1.0):
        super().__init__()
        self.encoder = Encoder(dim_in, dim_hidden, dim_latent)
        self.decoder = Decoder(dim_latent, dim_hidden, dim_in)
        self.beta = beta  # KL散度权重
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z
    
    def encode(self, x):
        """编码：返回潜在表示（使用均值，确定性）"""
        mu, logvar = self.encoder(x)
        return mu  # 推理时使用均值，避免随机性
    
    def decode(self, z):
        """解码：从潜在空间返回原始空间"""
        return self.decoder(z)
    
    def loss_function(self, x, recon, mu, logvar):
        """VAE损失 = 重建损失 + β * KL散度"""
        # 重建损失（MSE）
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL散度
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss


# ------------------------------
# 图增强VAE（可选：利用细胞邻域信息）
# ------------------------------
class GraphRegularizedVAE(VAE):
    """
    图正则化VAE：在VAE基础上添加图正则化
    - 保持完全可逆性
    - 利用细胞邻域信息提升表示质量
    """
    def __init__(self, dim_in, dim_hidden, dim_latent, beta=1.0, gamma=0.1):
        super().__init__(dim_in, dim_hidden, dim_latent, beta)
        self.gamma = gamma  # 图正则化权重
    
    def graph_regularization(self, z, adj):
        """
        图正则化：相邻细胞的潜在表示应该相似
        L_graph = sum_{i,j} A_{ij} * ||z_i - z_j||^2
        """
        # z: (n_cells, dim_latent)
        # adj: (n_cells, n_cells) 邻接矩阵
        
        diff = z.unsqueeze(0) - z.unsqueeze(1)  # (n, n, d)
        dist_sq = (diff ** 2).sum(dim=2)  # (n, n)
        
        # 只计算邻居之间的距离
        graph_loss = (adj * dist_sq).sum() / (adj.sum() + 1e-8)
        return graph_loss
    
    def loss_function_with_graph(self, x, recon, mu, logvar, z, adj):
        """带图正则化的损失"""
        total_loss, recon_loss, kl_loss = self.loss_function(x, recon, mu, logvar)
        
        graph_loss = self.graph_regularization(z, adj)
        total_loss = total_loss + self.gamma * graph_loss
        
        return total_loss, recon_loss, kl_loss, graph_loss


# ------------------------------
# 训练函数
# ------------------------------
def train_vae(model, X, adj=None, n_epochs=200, batch_size=512, lr=1e-3, device='cuda'):
    """训练VAE模型"""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    n_samples = X.shape[0]
    use_graph = adj is not None and isinstance(model, GraphRegularizedVAE)
    
    model.train()
    history = {'total': [], 'recon': [], 'kl': [], 'graph': []}
    
    for epoch in range(n_epochs):
        # 随机打乱数据
        perm = torch.randperm(n_samples)
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_graph = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            x_batch = X[idx].to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar, z = model(x_batch)
            
            if use_graph:
                adj_batch = adj[idx][:, idx].to(device)
                loss, recon_loss, kl_loss, graph_loss = model.loss_function_with_graph(
                    x_batch, recon, mu, logvar, z, adj_batch
                )
                total_graph += graph_loss.item()
            else:
                loss, recon_loss, kl_loss = model.loss_function(x_batch, recon, mu, logvar)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # 记录历史
        history['total'].append(total_loss / n_batches)
        history['recon'].append(total_recon / n_batches)
        history['kl'].append(total_kl / n_batches)
        if use_graph:
            history['graph'].append(total_graph / n_batches)
        
        if epoch % 20 == 0:
            msg = f"Epoch {epoch}: Loss={total_loss/n_batches:.4f}, Recon={total_recon/n_batches:.4f}, KL={total_kl/n_batches:.4f}"
            if use_graph:
                msg += f", Graph={total_graph/n_batches:.4f}"
            print(msg)
        
        torch.cuda.empty_cache()
    
    return history


# ------------------------------
# 主函数：创建VAE潜在表示
# ------------------------------
def create_vae_representation(
    adata, 
    dim_latent=64, 
    dim_hidden=[512, 256],
    model_save_path=None, 
    n_epochs=200,
    batch_size=512,
    lr=1e-3,
    beta=0.5,  # KL权重，<1使其更像AE（更好重建）
    use_graph=False,
    k_neighbors=15,
    gamma=0.1
):
    """
    使用VAE对单细胞数据进行降维
    
    参数:
        adata: AnnData对象
        dim_latent: 潜在空间维度
        dim_hidden: 隐藏层维度列表
        model_save_path: 模型保存路径
        n_epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        beta: KL散度权重（越小重建越好）
        use_graph: 是否使用图正则化
        k_neighbors: kNN邻居数（use_graph=True时使用）
        gamma: 图正则化权重
    
    返回:
        updated_adata: 更新后的AnnData
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 提取表达数据
    expression_data = adata.X
    if sp.issparse(expression_data):
        expression_data = expression_data.toarray()
    
    n_cells, n_genes = expression_data.shape
    print(f"数据形状: {n_cells} cells × {n_genes} genes")
    
    # 2. 数据标准化（保存参数用于逆变换）
    scaler = StandardScaler()
    expr_normalized = scaler.fit_transform(expression_data)
    
    # 转为张量
    X = torch.FloatTensor(expr_normalized)
    
    # 3. 构建邻接矩阵（如果使用图正则化）
    adj = None
    if use_graph:
        print(f"构建kNN图 (k={k_neighbors})...")
        adj_sparse = kneighbors_graph(
            expr_normalized, n_neighbors=k_neighbors, 
            mode='connectivity', include_self=True, metric='cosine'
        )
        adj = torch.FloatTensor(adj_sparse.toarray())
    
    # 4. 初始化模型
    if use_graph:
        model = GraphRegularizedVAE(
            dim_in=n_genes,
            dim_hidden=dim_hidden,
            dim_latent=dim_latent,
            beta=beta,
            gamma=gamma
        )
    else:
        model = VAE(
            dim_in=n_genes,
            dim_hidden=dim_hidden,
            dim_latent=dim_latent,
            beta=beta
        )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. 训练
    print("开始训练VAE...")
    history = train_vae(model, X, adj, n_epochs, batch_size, lr, device)
    
    # 6. 获取潜在表示
    print("获取潜在表示...")
    model.eval()
    with torch.no_grad():
        X_device = X.to(device)
        # 分批处理避免内存溢出
        latent_list = []
        for i in range(0, n_cells, batch_size):
            batch = X_device[i:i+batch_size]
            z_batch = model.encode(batch)
            latent_list.append(z_batch.cpu())
        latent_representations = torch.cat(latent_list, dim=0).numpy()
    
    # 7. 验证重建质量
    print("\n验证重建质量...")
    with torch.no_grad():
        sample_idx = np.random.choice(n_cells, min(1000, n_cells), replace=False)
        x_sample = X[sample_idx].to(device)
        z_sample = model.encode(x_sample)
        recon_sample = model.decode(z_sample).cpu().numpy()
        x_sample_np = x_sample.cpu().numpy()
        
        mse = np.mean((recon_sample - x_sample_np) ** 2)
        corr = np.corrcoef(recon_sample.flatten(), x_sample_np.flatten())[0, 1]
        print(f"重建MSE: {mse:.6f}")
        print(f"重建相关性: {corr:.4f}")
    
    # 8. 保存模型
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'GraphRegularizedVAE' if use_graph else 'VAE',
            'dim_in': n_genes,
            'dim_hidden': dim_hidden,
            'dim_latent': dim_latent,
            'beta': beta,
            'gamma': gamma if use_graph else None,
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_,
            'gene_names': list(adata.var_names),
            'history': history
        }, model_save_path)
        print(f"模型已保存到: {model_save_path}")
    
    # 9. 创建更新后的AnnData
    updated_adata = ad.AnnData(
        X=latent_representations,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=[f'latent_{i}' for i in range(dim_latent)]),
        uns=adata.uns.copy() if adata.uns else {}
    )
    
    # 保存重要信息
    updated_adata.obsm['X_latent'] = latent_representations
    updated_adata.uns['vae_info'] = {
        'dim_latent': dim_latent,
        'dim_hidden': dim_hidden,
        'model_save_path': model_save_path,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'gene_names': list(adata.var_names),
        'reconstruction_mse': mse,
        'reconstruction_corr': corr
    }
    
    # 复制原始数据的obsm
    for key in adata.obsm.keys():
        if key != 'X_latent':
            updated_adata.obsm[key] = adata.obsm[key]
    
    return updated_adata, model


# ------------------------------
# 解码函数：从潜在空间返回原始空间
# ------------------------------
def decode_to_expression(latent_data, model_path, device=None):
    """
    从潜在表示解码回原始基因表达空间
    
    参数:
        latent_data: 潜在表示 (n_cells, dim_latent)，可以是numpy数组或AnnData
        model_path: 模型路径
        device: 计算设备
    
    返回:
        reconstructed_adata: 重建的AnnData对象
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    
    # 重建模型
    model_type = checkpoint.get('model_type', 'VAE')
    if model_type == 'GraphRegularizedVAE':
        model = GraphRegularizedVAE(
            dim_in=checkpoint['dim_in'],
            dim_hidden=checkpoint['dim_hidden'],
            dim_latent=checkpoint['dim_latent'],
            beta=checkpoint['beta'],
            gamma=checkpoint['gamma']
        )
    else:
        model = VAE(
            dim_in=checkpoint['dim_in'],
            dim_hidden=checkpoint['dim_hidden'],
            dim_latent=checkpoint['dim_latent'],
            beta=checkpoint['beta']
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 提取潜在表示
    if isinstance(latent_data, ad.AnnData):
        z = latent_data.obsm.get('X_latent', latent_data.X)
        obs = latent_data.obs.copy()
    else:
        z = latent_data
        obs = None
    
    # 解码
    print("解码潜在表示...")
    with torch.no_grad():
        z_tensor = torch.FloatTensor(z).to(device)
        
        # 分批解码
        batch_size = 1024
        recon_list = []
        for i in range(0, z.shape[0], batch_size):
            batch = z_tensor[i:i+batch_size]
            recon_batch = model.decode(batch)
            recon_list.append(recon_batch.cpu())
        
        reconstructed_normalized = torch.cat(recon_list, dim=0).numpy()
    
    # 逆标准化
    scaler_mean = checkpoint['scaler_mean']
    scaler_scale = checkpoint['scaler_scale']
    reconstructed_expression = reconstructed_normalized * scaler_scale + scaler_mean
    
    # 创建AnnData
    gene_names = checkpoint['gene_names']
    reconstructed_adata = ad.AnnData(
        X=reconstructed_expression,
        var=pd.DataFrame(index=gene_names)
    )
    
    if obs is not None:
        reconstructed_adata.obs = obs
    
    print(f"重建完成: {reconstructed_adata.shape}")
    return reconstructed_adata


# ------------------------------
# 验证函数：测试编码-解码的可逆性
# ------------------------------
def verify_reversibility(original_adata, model_path, n_samples=1000, device=None):
    """
    验证编码-解码的可逆性
    
    返回:
        metrics: 包含MSE、相关性等指标的字典
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型和参数
    checkpoint = torch.load(model_path, map_location=device)
    
    model_type = checkpoint.get('model_type', 'VAE')
    if model_type == 'GraphRegularizedVAE':
        model = GraphRegularizedVAE(
            dim_in=checkpoint['dim_in'],
            dim_hidden=checkpoint['dim_hidden'],
            dim_latent=checkpoint['dim_latent'],
            beta=checkpoint['beta'],
            gamma=checkpoint['gamma']
        )
    else:
        model = VAE(
            dim_in=checkpoint['dim_in'],
            dim_hidden=checkpoint['dim_hidden'],
            dim_latent=checkpoint['dim_latent'],
            beta=checkpoint['beta']
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 准备数据
    expression_data = original_adata.X
    if sp.issparse(expression_data):
        expression_data = expression_data.toarray()
    
    # 随机采样
    n_cells = expression_data.shape[0]
    sample_idx = np.random.choice(n_cells, min(n_samples, n_cells), replace=False)
    x_sample = expression_data[sample_idx]
    
    # 标准化
    scaler_mean = checkpoint['scaler_mean']
    scaler_scale = checkpoint['scaler_scale']
    x_normalized = (x_sample - scaler_mean) / scaler_scale
    
    # 编码-解码
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x_normalized).to(device)
        z = model.encode(x_tensor)
        x_recon = model.decode(z).cpu().numpy()
    
    # 逆标准化
    x_recon_original = x_recon * scaler_scale + scaler_mean
    
    # 计算指标
    mse_normalized = np.mean((x_recon - x_normalized) ** 2)
    mse_original = np.mean((x_recon_original - x_sample) ** 2)
    corr_normalized = np.corrcoef(x_recon.flatten(), x_normalized.flatten())[0, 1]
    corr_original = np.corrcoef(x_recon_original.flatten(), x_sample.flatten())[0, 1]
    
    # 每个基因的重建相关性
    gene_corrs = []
    for i in range(x_sample.shape[1]):
        if np.std(x_sample[:, i]) > 0 and np.std(x_recon_original[:, i]) > 0:
            corr = np.corrcoef(x_sample[:, i], x_recon_original[:, i])[0, 1]
            gene_corrs.append(corr)
    
    metrics = {
        'mse_normalized': mse_normalized,
        'mse_original': mse_original,
        'correlation_normalized': corr_normalized,
        'correlation_original': corr_original,
        'mean_gene_correlation': np.nanmean(gene_corrs),
        'median_gene_correlation': np.nanmedian(gene_corrs),
        'min_gene_correlation': np.nanmin(gene_corrs),
        'max_gene_correlation': np.nanmax(gene_corrs)
    }
    
    print("\n=== 重建质量验证 ===")
    print(f"MSE (标准化空间): {mse_normalized:.6f}")
    print(f"MSE (原始空间): {mse_original:.6f}")
    print(f"整体相关性 (标准化): {corr_normalized:.4f}")
    print(f"整体相关性 (原始): {corr_original:.4f}")
    print(f"基因平均相关性: {np.nanmean(gene_corrs):.4f}")
    print(f"基因中位数相关性: {np.nanmedian(gene_corrs):.4f}")
    
    return metrics


# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    # 1. 加载数据
    print("加载数据...")
    adata = sc.read('/home/lenovo/jora/merged_ALL_by_time6.h5ad')
    print(f"原始数据: {adata.shape}")
    
    # 2. 模型保存路径
    model_save_path = "/home/lenovo/jora/vae_model128.pth"
    
    # 3. 训练VAE并获取潜在表示
    updated_adata, model = create_vae_representation(
        adata,
        dim_latent=128,           # 潜在空间维度
        dim_hidden=[512, 256],    # 隐藏层
        model_save_path=model_save_path,
        n_epochs=300,             # 训练轮数
        batch_size=512,
        lr=1e-3,
        beta=0.1,                 # 较小的beta → 更好的重建
        use_graph=True,           # 使用图正则化
        k_neighbors=15,
        gamma=0.05
    )
    
    print(f"\n更新后的数据: {updated_adata.shape}")
    
    # 4. 验证可逆性
    print("\n" + "="*50)
    metrics = verify_reversibility(adata, model_save_path, n_samples=2000)
    
    # 5. 测试解码功能
    print("\n" + "="*50)
    print("测试解码功能...")
    reconstructed_adata = decode_to_expression(updated_adata, model_save_path)
    print(f"重建数据形状: {reconstructed_adata.shape}")
    print(f"原始基因名前5个: {list(adata.var_names[:5])}")
    print(f"重建基因名前5个: {list(reconstructed_adata.var_names[:5])}")
    
    # 6. 保存结果
    output_path = "/home/lenovo/jora/adata_vae_latent128.h5ad"
    updated_adata.write(output_path)
    print(f"\n潜在表示已保存到: {output_path}")
    
    # 7. 可选：保存重建数据
    recon_output_path = "/home/lenovo/jora/adata_reconstructed.h5ad"
    reconstructed_adata.write(recon_output_path)
    print(f"重建数据已保存到: {recon_output_path}")