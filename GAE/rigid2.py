import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import pandas as pd
from scipy.spatial import cKDTree

class H5ADSimpleRotationTransformer:
    def __init__(self):
        """
        初始化 h5ad 数据简单旋转变换器
        """
        self.transformed_coords = {}  # 保存每个时间点变换后的坐标
        self.transformation_params = {}  # 保存每个时间点的变换参数
        self.original_adata = None  # 原始 AnnData 对象
        self.transformed_adata = None  # 变换后的 AnnData 对象
        
    def load_h5ad(self, file_path):
        """
        加载 h5ad 文件
        """
        self.original_adata = ad.read_h5ad(file_path)
        print(f"已加载 h5ad 文件: {file_path}")
        print(f"数据形状: {self.original_adata.shape}")
        print(f"观察值 (obs) 列: {list(self.original_adata.obs.columns)}")
        print(f"obsm 键: {list(self.original_adata.obsm.keys())}")
        
    def extract_timepoints(self, time_column='time'):
        """
        从 obs 中提取时间点信息
        """
        if time_column not in self.original_adata.obs.columns:
            raise ValueError(f"列 '{time_column}' 不存在于 obs 中")
            
        timepoints = self.original_adata.obs[time_column].unique()
        timepoint_indices = {}
        
        for tp in timepoints:
            timepoint_indices[tp] = self.original_adata.obs[self.original_adata.obs[time_column] == tp].index
        
        print(f"找到的时间点: {list(timepoints)}")
        return timepoints, timepoint_indices
    
    def extract_coordinates(self, coord_key='spatial'):
        """
        从 obsm 或 obs 中提取坐标
        """
        coordinates = {}
        timepoints, timepoint_indices = self.extract_timepoints('time')
        
        # 检查坐标是否在 obsm 中
        if coord_key in self.original_adata.obsm.keys():
            print(f"从 obsm['{coord_key}'] 中提取坐标")
            for tp in timepoints:
                mask = self.original_adata.obs['time'] == tp
                coords = self.original_adata.obsm[coord_key][mask]
                coordinates[tp] = coords
                print(f"时间点 {tp}: {coords.shape[0]} 个细胞")
        elif 'spatial_x' in self.original_adata.obs.columns and 'spatial_y' in self.original_adata.obs.columns:
            print("从 obs['spatial_x'] 和 obs['spatial_y'] 中提取坐标")
            for tp in timepoints:
                mask = self.original_adata.obs['time'] == tp
                x_coords = self.original_adata.obs['spatial_x'][mask].values
                y_coords = self.original_adata.obs['spatial_y'][mask].values
                coordinates[tp] = np.column_stack((x_coords, y_coords))
                print(f"时间点 {tp}: {coordinates[tp].shape[0]} 个细胞")
        else:
            raise ValueError("未找到空间坐标数据")
        
        return coordinates
    
    def create_rotation_matrix(self, angle_degrees):
        """
        创建2D旋转矩阵
        
        参数:
            angle_degrees: 旋转角度（度）
            
        返回:
            2x2旋转矩阵
        """
        angle_rad = np.radians(angle_degrees)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        
        return rotation_matrix
    
    def apply_rotation_translation(self, coords, rotation_angle, translation_vector, rotation_center=None):
        """
        应用旋转和平移变换
        
        参数:
            coords: 原始坐标
            rotation_angle: 旋转角度（度）
            translation_vector: 平移向量 [tx, ty]
            rotation_center: 旋转中心，如果为None则使用坐标中心
            
        返回:
            变换后的坐标
        """
        if rotation_center is None:
            rotation_center = np.mean(coords, axis=0)
        
        # 创建旋转矩阵
        R = self.create_rotation_matrix(rotation_angle)
        
        # 应用变换：先旋转，再平移
        coords_centered = coords - rotation_center
        coords_rotated = coords_centered @ R.T
        coords_transformed = coords_rotated + rotation_center + translation_vector
        
        return coords_transformed, R
    
    def calculate_shape_similarity(self, coords1, coords2):
        """
        计算两个点集之间的形状相似性，使用最近邻距离
        
        参数:
            coords1, coords2: 两个点集的坐标
            
        返回:
            相似性分数（越小越好）
        """
        # 使用KD树计算最近邻距离
        tree1 = cKDTree(coords1)
        tree2 = cKDTree(coords2)
        
        # 计算从coords1到coords2的最近邻距离
        dist1, _ = tree2.query(coords1)
        dist2, _ = tree1.query(coords2)
        
        # 使用Hausdorff距离的近似
        similarity = np.mean(dist1) + np.mean(dist2)
        
        return similarity
    
    def estimate_simple_transformation(self, coords, reference_coords):
        """
        估计简单的变换参数（小角度旋转和平移）
        
        参数:
            coords: 当前时间点坐标
            reference_coords: 参考时间点坐标
            
        返回:
            旋转角度（度）和平移向量
        """
        # 计算中心
        center_current = np.mean(coords, axis=0)
        center_reference = np.mean(reference_coords, axis=0)
        
        # 计算平移（使中心对齐）
        translation = center_reference - center_current
        
        # 尝试几个小角度，找到最佳匹配
        best_angle = 0
        best_error = float('inf')
        
        test_angles = [-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10]  # 测试的小角度
        
        for angle in test_angles:
            coords_transformed, _ = self.apply_rotation_translation(
                coords, angle, translation, center_current
            )
            
            # 使用形状相似性而不是点对点距离
            error = self.calculate_shape_similarity(coords_transformed, reference_coords)
            
            if error < best_error:
                best_error = error
                best_angle = angle
        
        print(f"最佳旋转角度: {best_angle}°, 相似性误差: {best_error:.4f}")
        return best_angle, translation
    
    def align_coordinates_simple(self, coordinates, reference_time):
        """
        使用简单方法对齐坐标（小角度旋转+平移）
        
        参数:
            coordinates: 每个时间点的坐标字典
            reference_time: 参考时间点
            
        返回:
            对齐后的坐标字典
        """
        if reference_time not in coordinates:
            raise ValueError(f"参考时间点 {reference_time} 不存在")
        
        aligned_coords = {}
        transformation_params = {}
        
        # 参考时间点不需要变换
        aligned_coords[reference_time] = coordinates[reference_time]
        transformation_params[reference_time] = {
            'rotation_angle': 0,
            'translation': np.zeros(2),
            'rotation_center': np.mean(coordinates[reference_time], axis=0)
        }
        
        for time_point, coords in coordinates.items():
            if time_point == reference_time:
                continue
                
            print(f"对齐时间点 {time_point} 到参考时间点 {reference_time}")
            
            # 估计变换参数
            rotation_angle, translation = self.estimate_simple_transformation(
                coords, coordinates[reference_time]
            )
            
            # 应用变换
            rotation_center = np.mean(coords, axis=0)
            aligned_coords[time_point], rotation_matrix = self.apply_rotation_translation(
                coords, rotation_angle, translation, rotation_center
            )
            
            # 保存参数
            transformation_params[time_point] = {
                'rotation_angle': rotation_angle,
                'translation': translation,
                'rotation_center': rotation_center,
                'rotation_matrix': rotation_matrix
            }
            
            self.transformation_params[time_point] = transformation_params[time_point]
        
        return aligned_coords
    
    def transform_h5ad(self, time_column='time', coord_key='spatial', reference_time=None):
        """
        对 h5ad 数据应用简单旋转变换
        
        参数:
            time_column: 时间信息列名
            coord_key: 坐标键名
            reference_time: 参考时间点，如果为None则使用第一个时间点
            
        返回:
            变换后的 AnnData 对象
        """
        # 提取时间点和坐标
        timepoints, timepoint_indices = self.extract_timepoints(time_column)
        coordinates = self.extract_coordinates(coord_key)
        
        # 设置参考时间点
        if reference_time is None:
            reference_time = timepoints[0]
        print(f"使用时间点 {reference_time} 作为参考")
        
        # 计算并应用变换
        aligned_coords = self.align_coordinates_simple(coordinates, reference_time)
        
        # 创建变换后的 AnnData 对象
        self.transformed_adata = self.original_adata.copy()
        
        # 创建新的坐标数组
        if coord_key in self.original_adata.obsm.keys():
            new_coords = np.zeros_like(self.original_adata.obsm[coord_key])
            for time_point in timepoints:
                mask = self.original_adata.obs[time_column] == time_point
                new_coords[mask] = aligned_coords[time_point]
            
            self.transformed_adata.obsm[f'{coord_key}_aligned'] = new_coords
        else:
            self.transformed_adata.obs['spatial_x_aligned'] = 0.0
            self.transformed_adata.obs['spatial_y_aligned'] = 0.0
            
            for time_point in timepoints:
                mask = self.original_adata.obs[time_column] == time_point
                self.transformed_adata.obs.loc[mask, 'spatial_x_aligned'] = aligned_coords[time_point][:, 0]
                self.transformed_adata.obs.loc[mask, 'spatial_y_aligned'] = aligned_coords[time_point][:, 1]
        
        # 保存变换参数到 uns
        self.transformed_adata.uns['transformation_parameters'] = {}
        for time_point, params in self.transformation_params.items():
            self.transformed_adata.uns['transformation_parameters'][str(time_point)] = {
                'translation': params['translation'].tolist(),
                'rotation_angle': float(params['rotation_angle']),
                'rotation_matrix': params['rotation_matrix'].tolist(),
                'rotation_center': params['rotation_center'].tolist()
            }
        
        print(f"已完成简单坐标变换，参考时间点为: {reference_time}")
        return self.transformed_adata
    
    def save_transformed_h5ad(self, file_path):
        """
        保存变换后的 h5ad 文件
        """
        if self.transformed_adata is None:
            raise ValueError("没有可保存的变换后数据，请先运行 transform_h5ad()")
            
        self.transformed_adata.write(file_path)
        print(f"变换后的数据已保存到: {file_path}")
    
    def visualize(self, coord_type='aligned', color_by='time', save_path=None):
        """
        可视化变换结果
        """
        if self.transformed_adata is None:
            raise ValueError("没有可可视化的变换后数据，请先运行 transform_h5ad()")
            
        vis_adata = self.transformed_adata.copy()
        
        # 确定坐标
        if coord_type == 'aligned':
            if 'spatial_aligned' in vis_adata.obsm.keys():
                basis = 'spatial_aligned'
            elif 'spatial_x_aligned' in vis_adata.obs.columns and 'spatial_y_aligned' in vis_adata.obs.columns:
                x_coord = vis_adata.obs['spatial_x_aligned'].values
                y_coord = vis_adata.obs['spatial_y_aligned'].values
                vis_adata.obsm['X_spatial_aligned'] = np.column_stack((x_coord, y_coord))
                basis = 'spatial_aligned'
            else:
                raise ValueError("未找到对齐后的坐标数据")
        else:
            if 'spatial' in vis_adata.obsm.keys():
                basis = 'spatial'
            elif 'spatial_x' in vis_adata.obs.columns and 'spatial_y' in vis_adata.obs.columns:
                x_coord = vis_adata.obs['spatial_x'].values
                y_coord = vis_adata.obs['spatial_y'].values
                vis_adata.obsm['X_spatial'] = np.column_stack((x_coord, y_coord))
                basis = 'spatial'
            else:
                raise ValueError("未找到原始坐标数据")
        
        # 绘制散点图
        plt.figure(figsize=(12, 10))
        sc.pl.embedding(
            vis_adata, 
            basis=basis, 
            color=color_by,
            show=False,
            size=20
        )
        plt.title(f"空间坐标 {'对齐后' if coord_type == 'aligned' else '原始'} - 按 {color_by} 着色")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        plt.show()
    
    def plot_comparison(self, time_column='time', save_path=None):
        """
        绘制原始坐标和对齐后坐标的对比
        """
        if self.transformed_adata is None:
            raise ValueError("没有可可视化的变换后数据，请先运行 transform_h5ad()")
            
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 提取原始坐标
        if 'spatial' in self.transformed_adata.obsm.keys():
            orig_coords = self.transformed_adata.obsm['spatial']
            orig_x = orig_coords[:, 0]
            orig_y = orig_coords[:, 1]
        elif 'spatial_x' in self.transformed_adata.obs.columns and 'spatial_y' in self.transformed_adata.obs.columns:
            orig_x = self.transformed_adata.obs['spatial_x'].values
            orig_y = self.transformed_adata.obs['spatial_y'].values
        else:
            raise ValueError("未找到原始坐标数据")
        
        # 提取对齐后坐标
        if 'spatial_aligned' in self.transformed_adata.obsm.keys():
            aligned_coords = self.transformed_adata.obsm['spatial_aligned']
            aligned_x = aligned_coords[:, 0]
            aligned_y = aligned_coords[:, 1]
        elif 'spatial_x_aligned' in self.transformed_adata.obs.columns and 'spatial_y_aligned' in self.transformed_adata.obs.columns:
            aligned_x = self.transformed_adata.obs['spatial_x_aligned'].values
            aligned_y = self.transformed_adata.obs['spatial_y_aligned'].values
        else:
            raise ValueError("未找到对齐后的坐标数据")
        
        # 绘制原始坐标
        timepoints = self.transformed_adata.obs[time_column].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(timepoints)))
        
        for i, tp in enumerate(timepoints):
            mask = self.transformed_adata.obs[time_column] == tp
            axes[0].scatter(orig_x[mask], orig_y[mask], color=colors[i], label=tp, alpha=0.7, s=10)
        
        axes[0].set_title("原始坐标")
        axes[0].set_xlabel("X坐标")
        axes[0].set_ylabel("Y坐标")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        
        # 绘制对齐后坐标
        for i, tp in enumerate(timepoints):
            mask = self.transformed_adata.obs[time_column] == tp
            axes[1].scatter(aligned_x[mask], aligned_y[mask], color=colors[i], label=tp, alpha=0.7, s=10)
        
        axes[1].set_title("对齐后坐标（小角度旋转+平移）")
        axes[1].set_xlabel("X坐标")
        axes[1].set_ylabel("Y坐标")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存到: {save_path}")
        
        plt.show()
    
    def print_transformation_info(self):
        """打印变换信息"""
        print("\n变换信息:")
        for time_point, params in self.transformation_params.items():
            print(f"时间点 {time_point}:")
            print(f"  旋转角度: {params['rotation_angle']:.2f}°")
            print(f"  平移向量: ({params['translation'][0]:.4f}, {params['translation'][1]:.4f})")
            print(f"  旋转中心: ({params['rotation_center'][0]:.4f}, {params['rotation_center'][1]:.4f})")


# 示例使用
if __name__ == "__main__":
    # 初始化变换器
    transformer = H5ADSimpleRotationTransformer()
    
    # 加载 h5ad 文件
    transformer.load_h5ad("/media/lenovo/A06B2FA1620B6FCB/pythonProject/data/RTime.h5ad")
    
    # 应用变换，将所有时间点对齐到第一个时间点
    transformed_adata = transformer.transform_h5ad(
        time_column='time',
        coord_key='spatial',
        reference_time=None
    )
    
    # 可视化变换结果
    print("\n原始坐标可视化:")
    transformer.visualize(coord_type='original', color_by='time')
    
    print("\n对齐后坐标可视化:")
    transformer.visualize(coord_type='aligned', color_by='time')
    
    # 绘制对比图
    print("\n原始坐标与对齐后坐标对比:")
    transformer.plot_comparison(time_column='time')
    
    # 保存变换后的 h5ad 文件
    transformer.save_transformed_h5ad("/media/lenovo/A06B2FA1620B6FCB/pythonProject/data/RTime_simple_aligned.h5ad")
    
    # 打印变换信息
    transformer.print_transformation_info()
    
    # 检查变换后的数据
    print(f"\n变换后的数据形状: {transformed_adata.shape}")
    print(f"变换后的 obsm 键: {list(transformed_adata.obsm.keys())}")
    if 'transformation_parameters' in transformed_adata.uns:
        print(f"变换后的 uns 中的变换参数: {list(transformed_adata.uns['transformation_parameters'].keys())}")