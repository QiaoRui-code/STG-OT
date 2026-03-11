import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from skimage.metrics import structural_similarity as ssim
import squidpy as sq
import scanpy as sc
# 示例数据：真实坐标和预测坐标（二维点集）
#true_coords = np.random.rand(100, 2) * 100  # 真实坐标 (100个点)
#pred_coords = np.random.rand(100, 2) * 100  # 预测坐标 (100个点)
adata_pred = sc.read_h5ad("/home/lenovo/jora/data/results_final_struct/pred_20dpi.h5ad")
pred_coords = adata_pred.obsm['X_spatial_aligned']  # 形状 (n_cells, 2)
    #hift_value1 = 1000  
    #shift_value2 = 80
    #pred_coords[:,0]+=shift_value1
    #pred_coords[:,1]-=shift_value2
adata2 = sc.read_h5ad('/home/lenovo/jora/data/R5_filtered_latent.h5ad')
    #adata2=sc.read_h5ad('/media/lenovo/A06B2FA1620B6FCB/LU/moscot/adata_with_spatial.h5ad')
    
adata2
adata1_sub=adata2[adata2.obs['time']==0]
adata2_sub=adata2[adata2.obs['time']==1]
adata3_sub=adata2[adata2.obs['time']==2]
adata4_sub=adata2[adata2.obs['time']==3]
adata5_sub=adata2[adata2.obs['time']==4]
adata6_sub=adata2[adata2.obs['time']== 5]
adata_true=adata5_sub
    #adata_true = sc.read_h5ad("/media/lenovo/A06B2FA1620B6FCB/LU/moscot/adata4.h5ad")
true_coords = adata_true.obsm['spatial_aligned']
# 定义图像尺寸
img_size = (256, 256)  # 图像分辨率

def coords_to_density(coords, img_size):
    """将坐标转换为密度图"""
    x, y = coords[:, 0], coords[:, 1]
    
    # 初始化图像网格
    grid_x = np.linspace(11000,16000, img_size[1])  # 假设坐标范围[0, 100]
    grid_y = np.linspace(0, 4000, img_size[0])
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    
    # 使用核密度估计生成密度图
    kde = gaussian_kde(np.vstack([x, y]))
    density = kde(grid_points).reshape(img_size)
    return density / np.max(density)  # 归一化到[0,1]

# 生成真实和预测的密度图
true_density = coords_to_density(true_coords, img_size)
pred_density = coords_to_density(pred_coords, img_size)

# 计算SSIM（确保两幅图像尺寸相同）
ssim_value, ssim_image = ssim(
    true_density, 
    pred_density,
    full=True,           # 返回完整SSIM图像
    data_range=1.0,      # 数据范围[0,1]
    win_size=11,         # 滑动窗口大小
    gradient=False,      # 不计算梯度
    use_sample_covariance=True
)

print(f"SSIM = {ssim_value:.4f}")