import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import cKDTree, distance
from sklearn.neighbors import NearestNeighbors

# 设置中文支持


def compute_local_density(coords, k=10):
    """计算基于k近邻的局部密度权重"""
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k+1)
    avg_dists = np.mean(dists[:, 1:], axis=1)
    densities = 1 / (avg_dists + 1e-6)
    return densities / np.max(densities)

def improved_spatial_rmse(adata_pred, adata_true, resolution=1.0):
    """
    改进的空间RMSE计算，解决密度不匹配问题
    """
    pred_coords = adata_pred.obsm['X_spatial_aligned']
    true_coords = adata_true.obsm['spatial_aligned']
    
    # 密度自适应匹配
    nbrs = NearestNeighbors(n_neighbors=1).fit(true_coords)
    distances, indices = nbrs.kneighbors(pred_coords)
    matched_true_coords = true_coords[indices[:, 0]]
    
    # 计算局部密度权重 - 关键修复：为匹配点计算权重
    # 使用匹配的真实点坐标计算密度权重，而不是全部真实点
    density_weights = compute_local_density(matched_true_coords)
    
    # 计算加权距离
    dists = np.linalg.norm(pred_coords - matched_true_coords, axis=1)
    max_migration = 50.0 / resolution
    constrained_dists = np.minimum(dists, max_migration)
    
    # 确保形状一致
    assert constrained_dists.shape == density_weights.shape, \
        f"形状不匹配: 距离数组{constrained_dists.shape}, 权重数组{density_weights.shape}"
    
    # 计算加权RMSE
    weighted_mse = np.average(constrained_dists**2, weights=density_weights)
    weighted_rmse = np.sqrt(weighted_mse)
    
    return weighted_rmse, matched_true_coords

def calculate_avg_spacing(coords):
    dists = distance.pdist(coords)
    return np.median(dists)

# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    
    import squidpy as sq
    import scanpy as sc
    
    # 加载数据
    adata_pred = sc.read_h5ad("/home/lenovo/jora/data/results_final_struct/pred_20dpi.h5ad")
    print(adata_pred)
    pred_coords = adata_pred.obsm['X_spatial_aligned']  # 形状 (n_cells, 2)
    #shift_value1 = 180  
    #shift_value2 = -20
    #pred_coords[:,0]+=shift_value1
    #pred_coords[:,1]+=shift_value2
    #y_min, y_max = -100, -70
   # mask = (
   #(pred_coords[:, 1] >= y_min) &  # X轴下限
    #(pred_coords[:, 1] <= y_max)
#)

# 应用筛选
    #pred_coords = pred_coords[mask]
    #pred_coords[:,1]-=shift_value2
    #adata2 = sc.read_h5ad('/media/lenovo/A06B2FA1620B6FCB/LU/mouse/Lvariable_genes.h5ad')
    #adata2
    #adata1_sub=adata2[adata2.obs['time']=='3.3hpf']
    #adata2_sub=adata2[adata2.obs['time']=='5.25hpf']
    #adata3_sub=adata2[adata2.obs['time']=='10hpf']
    #adata4_sub=adata2[adata2.obs['time']=='12hpf']
    #adata5_sub=adata2[adata2.obs['time']=='18hpf']
    #adata6_sub=adata2[adata2.obs['time']=='24hpf']
    #adata_true=adata5_sub
    adata2 = sc.read_h5ad('/home/lenovo/jora/data/R5_filtered_latent.h5ad')
    #adata2=sc.read_h5ad('/media/lenovo/A06B2FA1620B6FCB/LU/moscot/adata_with_spatial.h5ad')
    
    print(adata2)
    adata1_sub=adata2[adata2.obs['time']==0]
    adata2_sub=adata2[adata2.obs['time']==1]
    adata3_sub=adata2[adata2.obs['time']==2]
    adata4_sub=adata2[adata2.obs['time']==3]
    adata5_sub=adata2[adata2.obs['time']==4]
    adata6_sub=adata2[adata2.obs['time']== 5]
    adata_true=adata5_sub
    true_coords = adata_true.obsm['spatial_aligned']

    # 计算平均间距
    avg_spacing = calculate_avg_spacing(true_coords)
    print(f"average spacing: {avg_spacing:.2f}")
    
    # 计算改进的RMSE
    rmse, matched_coords = improved_spatial_rmse(
        adata_pred, 
        adata_true,
        resolution=0.5
    )
    
    print(f"RMSE: {rmse:.2f} ")
    
    # 可视化结果
    plt.figure(figsize=(14, 6))
    
    # 真实分布
    plt.subplot(131)
    plt.scatter(true_coords[:, 0], true_coords[:, 1], s=5, c='blue', alpha=0.6)
    plt.title('true spatial distribution')
    plt.axis('equal')
    
    # 预测分布
    plt.subplot(132)
    plt.scatter(pred_coords[:, 0], pred_coords[:, 1], s=5, c='red', alpha=0.6)
    plt.title('predicted spatial distribution')
    plt.axis('equal')
    
    # 匹配结果
    plt.subplot(133)
    for i in range(len(pred_coords)):
        plt.plot([pred_coords[i, 0], matched_coords[i, 0]],
                 [pred_coords[i, 1], matched_coords[i, 1]], 
                 'k-', alpha=0.02, linewidth=0.5)
    plt.scatter(pred_coords[:, 0], pred_coords[:, 1], s=5, c='red', alpha=0.6, label='predicted')
    plt.scatter(matched_coords[:, 0], matched_coords[:, 1], s=5, c='blue', alpha=0.8, label='matched')
    plt.title('spatial distribution matching')
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('spatial_rmse_comparisonL4.png', dpi=300)
    plt.show()