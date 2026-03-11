import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import math
import warnings
from datetime import datetime

# 清除环境变量
if 'PYTORCH_CUDA_ALLOC_CONF' in os.environ:
    del os.environ['PYTORCH_CUDA_ALLOC_CONF']
warnings.filterwarnings("ignore")

# ==========================================
# 1. 模型结构
# ==========================================

class SinusoidalTimeEmbeddings(nn.Module):
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
            SinusoidalTimeEmbeddings(time_emb_dim),
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
# 2. 辅助函数
# ==========================================
def get_soft_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes)
    counts = np.maximum(counts, 1) 
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * num_classes
    return torch.FloatTensor(weights)

def load_and_prep_data(datapath):
    print(f"读取数据: {datapath}")
    adata = sc.read_h5ad(datapath)
    
    if 'X_latent' in adata.obsm:
        print("使用 X_latent")
        gene_expr = adata.obsm['X_latent']
        
    elif 'X_pca' in adata.obsm:
        print("使用 X_pca")
        gene_expr = adata.obsm['X_pca']
    else:
        print("使用 adata.X")
        gene_expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    spatial = adata.obsm['spatial_aligned'] if 'spatial_aligned' in adata.obsm else adata.obsm['spatial']
    
    time = adata.obs['time']
    if hasattr(time, 'cat'):
        time = time.cat.codes.values.astype(float)
    else:
        time = pd.to_numeric(time, errors='coerce').values
        
    le = LabelEncoder()
    y = le.fit_transform(adata.obs['Annotation'])
    
    return gene_expr, spatial, time, y, le

# ==========================================
# 3. 训练流程 (包含保存逻辑)
# ==========================================
def train_balanced_model(datapath, save_dir="./saved_models"):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"balanced_model_{timestamp}.pth")

    # 1. 加载
    gene_expr, spatial, time, y, le = load_and_prep_data(datapath)
    num_classes = len(le.classes_)
    print(f"输入特征维度: {gene_expr.shape[1]}")
    print(f"类别数量: {num_classes}")
    
    # 2. 划分
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Scaler
    g_scaler = StandardScaler().fit(gene_expr[train_idx])
    s_scaler = StandardScaler().fit(spatial[train_idx])
    t_scaler = StandardScaler().fit(time[train_idx].reshape(-1, 1))
    
    # --- 关键：打包预处理器 ---
    preprocessors = {
        'le': le,
        'gs': g_scaler,
        'ss': s_scaler,
        'ts': t_scaler
    }
    
    # 4. Tensor
    data = {
        'x_train': torch.FloatTensor(g_scaler.transform(gene_expr[train_idx])),
        'x_test': torch.FloatTensor(g_scaler.transform(gene_expr[test_idx])),
        'z_train': torch.FloatTensor(s_scaler.transform(spatial[train_idx])),
        'z_test': torch.FloatTensor(s_scaler.transform(spatial[test_idx])),
        't_train': torch.FloatTensor(t_scaler.transform(time[train_idx].reshape(-1, 1))),
        't_test': torch.FloatTensor(t_scaler.transform(time[test_idx].reshape(-1, 1))),
        'y_train': torch.LongTensor(y[train_idx]),
        'y_test': torch.LongTensor(y[test_idx])
    }
    
    # 5. 权重
    class_weights = get_soft_class_weights(y[train_idx], num_classes)
    
    # 6. 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BalancedClassifier(
        gene_dim=data['x_train'].shape[1],
        spatial_dim=data['z_train'].shape[1],
        hidden_dims=[512, 256],
        output_dim=num_classes
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data['x_train'], data['z_train'], data['t_train'], data['y_train']),
        batch_size=256, shuffle=True
    )
    
    print(f"=== 开始平衡训练 (模型将保存至: {save_path}) ===")
    best_acc = 0
    epochs = 120
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for bx, bz, bt, by in loader:
            bx, bz, bt, by = bx.to(device), bz.to(device), bt.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx, bz, bt)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = torch.max(logits, 1)
            correct += (pred == by).sum().item()
            total += by.size(0)
            
        train_acc = 100 * correct / total
        
        # Eval
        model.eval()
        with torch.no_grad():
            tx, tz, tt, ty = data['x_test'].to(device), data['z_test'].to(device), data['t_test'].to(device), data['y_test'].to(device)
            out = model(tx, tz, tt)
            _, t_pred = torch.max(out, 1)
            test_acc = 100 * (t_pred == ty).sum().item() / ty.size(0)
            
        scheduler.step(test_acc)
        
        # --- 关键：保存最佳模型 ---
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'preprocessors': preprocessors, # 这里保存了所有需要的转换器
                'best_acc': best_acc,
                'gene_dim': data['x_train'].shape[1], # 保存维度信息以防万一
                'spatial_dim': data['z_train'].shape[1]
            }, save_path)
            print(f"Epoch {epoch+1:03d}: 测试准确率提升至 {test_acc:.2f}% -> 模型已保存")
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {train_loss/len(loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
            
    print(f"\n训练完成! 最佳测试准确率: {best_acc:.2f}%")
    print(f"最佳模型文件: {save_path}")
    
    # 最终报告
    # 加载最佳模型参数来生成报告，确保报告对应的是保存的模型
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        tx, tz, tt, ty = data['x_test'].to(device), data['z_test'].to(device), data['t_test'].to(device), data['y_test'].to(device)
        logits = model(tx, tz, tt)
        _, preds = torch.max(logits, 1)
        
        _, top3 = logits.topk(3, 1, largest=True, sorted=True)
        top3_correct = sum([ty[i] in top3[i] for i in range(len(ty))])
        print(f"最终最佳模型的 Top-3 准确率: {100*top3_correct/len(ty):.2f}%")

    print("\n分类报告:")
    print(classification_report(ty.cpu().numpy(), preds.cpu().numpy(), digits=3))

if __name__ == "__main__":
    path = "/home/lenovo/jora/data/R5_filtered_latent.h5ad"
    if os.path.exists(path):
        train_balanced_model(path)