import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.optimize import minimize
import anndata as ad
import scanpy as sc
import pandas as pd

class H5ADTranslationTransformer:
    def __init__(self):
        """
        初始化 h5ad 数据平移变换器
        """
        self.transformed_coords = {}  # 保存每个时间点变换后的坐标
        self.translation_vectors = {}  # 保存每个时间点的平移向量
        self.original_adata = None  # 原始 AnnData 对象
        self.transformed_adata = None  # 变换后的 AnnData 对象
        
    def load_h5ad(self, file_path):
        """
        加载 h5ad 文件
        
        参数:
            file_path: h5ad 文件路径
        """
        self.original_adata = ad.read_h5ad(file_path)
        print(f"已加载 h5ad 文件: {file_path}")
        print(f"数据形状: {self.original_adata.shape}")
        print(f"观察值 (obs) 列: {list(self.original_adata.obs.columns)}")
        print(f"obsm 键: {list(self.original_adata.obsm.keys())}")
        
    def extract_timepoints(self, time_column='time'):
        """
        从 obs 中提取时间点信息
        
        参数:
            time_column: 包含时间信息的列名
            
        返回:
            时间点列表和每个时间点的索引
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
        
        参数:
            coord_key: obsm 中的坐标键名或 obs 中的坐标列前缀
            
        返回:
            每个时间点的坐标字典
        """
        coordinates = {}
        timepoints, timepoint_indices = self.extract_timepoints('time')
        
        # 检查坐标是否在 obsm 中
        if coord_key in self.original_adata.obsm.keys():
            print(f"从 obsm['{coord_key}'] 中提取坐标")
            for tp, indices in timepoint_indices.items():
                # 获取该时间点的坐标
                mask = self.original_adata.obs['time'] == tp
                coords = self.original_adata.obsm[coord_key][mask]
                coordinates[tp] = coords
        # 检查坐标是否在 obs 中 (spatial_x 和 spatial_y)
        elif 'spatial_x' in self.original_adata.obs.columns and 'spatial_y' in self.original_adata.obs.columns:
            print("从 obs['spatial_x'] 和 obs['spatial_y'] 中提取坐标")
            for tp, indices in timepoint_indices.items():
                # 获取该时间点的坐标
                mask = self.original_adata.obs['time'] == tp
                x_coords = self.original_adata.obs['spatial_x'][mask].values
                y_coords = self.original_adata.obs['spatial_y'][mask].values
                coordinates[tp] = np.column_stack((x_coords, y_coords))
        else:
            raise ValueError("未找到空间坐标数据")
        
        return coordinates
    
    def apply_translation(self, coordinates, translation_vector, time_point):
        """
        应用平移变换到坐标
        
        参数:
            coordinates: 原始坐标
            translation_vector: 平移向量 [tx, ty]
            time_point: 时间点标识
            
        返回:
            平移后的坐标
        """
        # 应用平移: p' = p + t
        translated_coords = coordinates + translation_vector
        
        # 保存变换后的坐标和平移向量
        self.transformed_coords[time_point] = translated_coords.copy()
        self.translation_vectors[time_point] = translation_vector.copy()
        
        return translated_coords
    
    def calculate_alignment_translations(self, coordinates, reference_time):
        """
        计算将所有时间点对齐到参考时间点所需的平移向量
        
        参数:
            coordinates: 每个时间点的坐标字典
            reference_time: 参考时间点
            
        返回:
            平移向量字典 {time_point: translation_vector}
        """
        if reference_time not in coordinates:
            raise ValueError(f"参考时间点 {reference_time} 不存在")
            
        # 计算参考时间点的中心
        ref_center = np.mean(coordinates[reference_time], axis=0)
        
        translation_vectors = {}
        
        for time_point, coords in coordinates.items():
            if time_point == reference_time:
                # 参考时间点不需要平移
                translation_vectors[time_point] = np.zeros(2)
            else:
                # 计算当前时间点的中心
                current_center = np.mean(coords, axis=0)
                # 计算使两个中心对齐的平移向量
                translation_vectors[time_point] = ref_center - current_center
        
        return translation_vectors
    
    def align_coordinates(self, coordinates, reference_time):
        """
        将所有时间点的坐标对齐到参考时间点
        
        参数:
            coordinates: 每个时间点的坐标字典
            reference_time: 参考时间点
            
        返回:
            对齐后的坐标字典
        """
        # 计算平移向量
        translations = self.calculate_alignment_translations(coordinates, reference_time)
        
        # 应用平移
        aligned_coords = {}
        for time_point, coords in coordinates.items():
            aligned_coords[time_point] = coords + translations[time_point]
            
        return aligned_coords
    
    def transform_h5ad(self, time_column='time', coord_key='spatial', reference_time=None):
        """
        对 h5ad 数据应用平移变换
        
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
        
        # 计算并应用平移
        aligned_coords = self.align_coordinates(coordinates, reference_time)
        
        # 创建变换后的 AnnData 对象
        self.transformed_adata = self.original_adata.copy()
        
        # 创建新的坐标数组
        if coord_key in self.original_adata.obsm.keys():
            # 如果坐标在 obsm 中
            new_coords = np.zeros_like(self.original_adata.obsm[coord_key])
            for time_point, indices in timepoint_indices.items():
                mask = self.original_adata.obs[time_column] == time_point
                new_coords[mask] = aligned_coords[time_point]
            
            # 将变换后的坐标添加到 obsm
            self.transformed_adata.obsm[f'{coord_key}_aligned'] = new_coords
        else:
            # 如果坐标在 obs 中
            self.transformed_adata.obs['spatial_x_aligned'] = 0.0
            self.transformed_adata.obs['spatial_y_aligned'] = 0.0
            
            for time_point, indices in timepoint_indices.items():
                mask = self.original_adata.obs[time_column] == time_point
                self.transformed_adata.obs.loc[mask, 'spatial_x_aligned'] = aligned_coords[time_point][:, 0]
                self.transformed_adata.obs.loc[mask, 'spatial_y_aligned'] = aligned_coords[time_point][:, 1]
        
        # 保存平移向量到 uns
        self.transformed_adata.uns['translation_vectors'] = {
            str(k): v.tolist() for k, v in self.translation_vectors.items()
        }
        
        print(f"已完成坐标平移变换，参考时间点为: {reference_time}")
        return self.transformed_adata
    
    def save_transformed_h5ad(self, file_path):
        """
        保存变换后的 h5ad 文件
        
        参数:
            file_path: 输出文件路径
        """
        if self.transformed_adata is None:
            raise ValueError("没有可保存的变换后数据，请先运行 transform_h5ad()")
            
        self.transformed_adata.write(file_path)
        print(f"变换后的数据已保存到: {file_path}")
    
    def visualize(self, coord_type='aligned', color_by='time'):
        """
        可视化变换结果
        
        参数:
            coord_type: 坐标类型 ('original' 或 'aligned')
            color_by: 着色依据的列名
        """
        if self.transformed_adata is None:
            raise ValueError("没有可可视化的变换后数据，请先运行 transform_h5ad()")
            
        # 创建数据副本用于可视化
        vis_adata = self.transformed_adata.copy()
        
        # 确定坐标
        if coord_type == 'aligned':
            if 'spatial_aligned' in vis_adata.obsm.keys():
                basis = 'spatial_aligned'
            elif 'spatial_x_aligned' in vis_adata.obs.columns and 'spatial_y_aligned' in vis_adata.obs.columns:
                # 从 obs 中提取对齐后的坐标并添加到 obsm
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
                # 从 obs 中提取原始坐标并添加到 obsm
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
            show=False
        )
        plt.title(f"空间坐标 {'对齐后' if coord_type == 'aligned' else '原始'} - 按 {color_by} 着色")
        plt.tight_layout()
        plt.show()
    
    def plot_centers(self, coord_type='aligned', time_column='time'):
        """
        绘制每个时间点的中心点变化
        
        参数:
            coord_type: 坐标类型 ('original' 或 'aligned')
            time_column: 时间信息列名
        """
        if self.transformed_adata is None:
            raise ValueError("没有可可视化的变换后数据，请先运行 transform_h5ad()")
            
        # 提取坐标
        if coord_type == 'aligned':
            if 'spatial_aligned' in self.transformed_adata.obsm.keys():
                coords = self.transformed_adata.obsm['spatial_aligned']
            elif 'spatial_x_aligned' in self.transformed_adata.obs.columns and 'spatial_y_aligned' in self.transformed_adata.obs.columns:
                x_coord = self.transformed_adata.obs['spatial_x_aligned'].values
                y_coord = self.transformed_adata.obs['spatial_y_aligned'].values
                coords = np.column_stack((x_coord, y_coord))
            else:
                raise ValueError("未找到对齐后的坐标数据")
        else:
            if 'spatial' in self.transformed_adata.obsm.keys():
                coords = self.transformed_adata.obsm['spatial']
            elif 'spatial_x' in self.transformed_adata.obs.columns and 'spatial_y' in self.transformed_adata.obs.columns:
                x_coord = self.transformed_adata.obs['spatial_x'].values
                y_coord = self.transformed_adata.obs['spatial_y'].values
                coords = np.column_stack((x_coord, y_coord))
            else:
                raise ValueError("未找到原始坐标数据")
        
        # 计算每个时间点的中心
        centers = {}
        timepoints = self.transformed_adata.obs[time_column].unique()
        
        for tp in timepoints:
            mask = self.transformed_adata.obs[time_column] == tp
            centers[tp] = np.mean(coords[mask], axis=0)
        
        # 绘制中心点
        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(centers)))
        
        for i, (time_point, center) in enumerate(centers.items()):
            plt.scatter(center[0], center[1], color=colors[i], s=200, label=time_point, alpha=0.8)
        
        plt.title(f"{'对齐后' if coord_type == 'aligned' else '原始'}各时间点中心点位置")
        plt.xlabel("X坐标")
        plt.ylabel("Y坐标")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()
    
    def plot_comparison(self, time_column='time'):
        """
        绘制原始坐标和对齐后坐标的对比
        
        参数:
            time_column: 时间信息列名
        """
        if self.transformed_adata is None:
            raise ValueError("没有可可视化的变换后数据，请先运行 transform_h5ad()")
            
        # 创建子图
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
        
        axes[1].set_title("对齐后坐标")
        axes[1].set_xlabel("X坐标")
        axes[1].set_ylabel("Y坐标")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')
        
        plt.tight_layout()
        plt.show()


# 示例使用
if __name__ == "__main__":
    # 初始化变换器
    transformer = H5ADTranslationTransformer()
    
    # 加载 h5ad 文件
    transformer.load_h5ad("/media/lenovo/A06B2FA1620B6FCB/pythonProject/data/RTime.h5ad")
    
    # 应用变换，将所有时间点对齐到第一个时间点
    transformed_adata = transformer.transform_h5ad(
        time_column='time',  # 使用 'time' 列
        coord_key='spatial',  # 使用 'spatial' 键
        reference_time=None  # 使用第一个时间点作为参考
    )
    
    # 可视化变换结果
    print("\n原始坐标可视化:")
    transformer.visualize(coord_type='original', color_by='time')
    
    print("\n对齐后坐标可视化:")
    transformer.visualize(coord_type='aligned', color_by='time')
    
    # 绘制对比图
    print("\n原始坐标与对齐后坐标对比:")
    transformer.plot_comparison(time_column='time')
    
    # 绘制中心点
    print("\n原始坐标中心点:")
    transformer.plot_centers(coord_type='original', time_column='time')
    
    print("\n对齐后坐标中心点:")
    transformer.plot_centers(coord_type='aligned', time_column='time')
    
    # 保存变换后的 h5ad 文件
    transformer.save_transformed_h5ad("/media/lenovo/A06B2FA1620B6FCB/pythonProject/data/RTime_aligned.h5ad")
    
    # 打印变换信息
    print("\n变换信息:")
    for time_point, vector in transformer.translation_vectors.items():
        print(f"时间点 {time_point}: 平移向量 ({vector[0]:.4f}, {vector[1]:.4f})")
    
    # 检查变换后的数据
    print(f"\n变换后的数据形状: {transformed_adata.shape}")
    print(f"变换后的 obsm 键: {list(transformed_adata.obsm.keys())}")
    if 'translation_vectors' in transformed_adata.uns:
        print(f"变换后的 uns 中的平移向量: {list(transformed_adata.uns['translation_vectors'].keys())}")