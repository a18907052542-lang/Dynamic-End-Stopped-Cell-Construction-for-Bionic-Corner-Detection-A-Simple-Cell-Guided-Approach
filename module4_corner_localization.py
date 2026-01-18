"""
Module 4: Corner Localization Based on Hierarchical Clustering
基于简单细胞与端抑制细胞协同处理的仿生弯角检测方法 - 角点定位模块

This module implements hierarchical clustering and dual-cluster centroid fusion
for precise corner localization from end-stopped cell response points.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict


class CornerLocalizer:
    """
    角点定位器 - 基于层次聚类的双簇质心融合
    
    从端抑制细胞的显著响应点中提取角点的精确坐标
    """
    
    def __init__(self,
                 distance_threshold: float = 15.0,
                 min_cluster_size: int = 3,
                 max_pairing_distance: float = 40.0,
                 linkage_method: str = 'average'):
        """
        初始化角点定位器
        
        Parameters:
        -----------
        distance_threshold : float
            聚类距离阈值 T_d (像素)
        min_cluster_size : int
            最小簇点数 N_min
        max_pairing_distance : float
            最大配对距离 D_max (像素)
        linkage_method : str
            链接方法 ('average', 'single', 'complete')
        """
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.max_pairing_distance = max_pairing_distance
        self.linkage_method = linkage_method
        
    def hierarchical_clustering(self, 
                                 points: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        执行层次聚类
        
        使用自底向上的凝聚策略，以欧氏距离作为相似性度量
        
        Parameters:
        -----------
        points : np.ndarray
            响应点坐标 (N, 2)
            
        Returns:
        --------
        tuple
            (cluster_labels, cluster_info)
        """
        if len(points) < 2:
            return np.zeros(len(points), dtype=int), {}
        
        # 计算距离矩阵
        distances = pdist(points, metric='euclidean')
        
        # 层次聚类
        Z = linkage(distances, method=self.linkage_method)
        
        # 根据距离阈值切割树
        cluster_labels = fcluster(Z, t=self.distance_threshold, criterion='distance')
        
        # 统计聚类信息
        cluster_info = self._compute_cluster_info(points, cluster_labels)
        
        return cluster_labels, cluster_info
    
    def _compute_cluster_info(self, 
                               points: np.ndarray, 
                               labels: np.ndarray) -> Dict:
        """
        计算聚类簇的统计信息
        
        Parameters:
        -----------
        points : np.ndarray
            点坐标
        labels : np.ndarray
            聚类标签
            
        Returns:
        --------
        dict
            各簇的统计信息
        """
        cluster_info = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            cluster_points = points[mask]
            
            if len(cluster_points) >= self.min_cluster_size:
                # 计算质心
                centroid = np.mean(cluster_points, axis=0)
                
                cluster_info[label] = {
                    'centroid': centroid,
                    'size': len(cluster_points),
                    'points': cluster_points,
                    'std': np.std(cluster_points, axis=0) if len(cluster_points) > 1 else np.array([0, 0])
                }
        
        return cluster_info
    
    def find_cluster_pairs(self, cluster_info: Dict) -> List[Tuple[int, int]]:
        """
        寻找配对的聚类簇
        
        分析聚类簇的空间分布模式，识别配对的簇对
        
        Parameters:
        -----------
        cluster_info : dict
            聚类簇信息
            
        Returns:
        --------
        list
            配对的簇标签列表 [(label1, label2), ...]
        """
        pairs = []
        labels = list(cluster_info.keys())
        used = set()
        
        for i, label1 in enumerate(labels):
            if label1 in used:
                continue
            
            best_pair = None
            min_distance = self.max_pairing_distance
            
            for label2 in labels[i+1:]:
                if label2 in used:
                    continue
                
                # 计算两个簇质心之间的距离
                centroid1 = cluster_info[label1]['centroid']
                centroid2 = cluster_info[label2]['centroid']
                distance = np.linalg.norm(centroid1 - centroid2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_pair = label2
            
            if best_pair is not None:
                pairs.append((label1, best_pair))
                used.add(label1)
                used.add(best_pair)
        
        return pairs
    
    def compute_corner_position(self,
                                 cluster_info: Dict,
                                 pair: Tuple[int, int],
                                 weight_ratio: float = 0.5) -> np.ndarray:
        """
        计算角点位置
        
        通过双簇质心融合确定角点坐标
        p_corner = (1/2) * (centroid1 + centroid2)
        
        Parameters:
        -----------
        cluster_info : dict
            聚类簇信息
        pair : tuple
            配对的簇标签
        weight_ratio : float
            融合权重（0.5为等权重）
            
        Returns:
        --------
        np.ndarray
            角点坐标 [y, x]
        """
        label1, label2 = pair
        centroid1 = cluster_info[label1]['centroid']
        centroid2 = cluster_info[label2]['centroid']
        
        # 加权融合
        corner_pos = weight_ratio * centroid1 + (1 - weight_ratio) * centroid2
        
        return corner_pos
    
    def localize_corners(self,
                         significant_points: np.ndarray,
                         responses: np.ndarray = None) -> List[Dict]:
        """
        执行完整的角点定位流程
        
        Parameters:
        -----------
        significant_points : np.ndarray
            显著响应点坐标 (N, 2)
        responses : np.ndarray
            响应值（可选，用于加权）
            
        Returns:
        --------
        list
            检测到的角点列表，每个元素包含位置和相关信息
        """
        if len(significant_points) < 2:
            if len(significant_points) == 1:
                return [{
                    'position': significant_points[0],
                    'cluster_labels': [0],
                    'confidence': 1.0,
                    'type': 'single_point'
                }]
            return []
        
        # 使用响应值加权的质心计算
        if responses is not None and len(responses) == len(significant_points):
            weights = responses / responses.sum() if responses.sum() > 0 else np.ones(len(responses)) / len(responses)
            weighted_center = np.average(significant_points, axis=0, weights=weights)
        else:
            weighted_center = np.mean(significant_points, axis=0)
        
        # 层次聚类用于识别边缘方向
        labels, cluster_info = self.hierarchical_clustering(significant_points)
        
        corners = []
        
        if len(cluster_info) >= 2:
            # 寻找配对簇
            pairs = self.find_cluster_pairs(cluster_info)
            
            for pair in pairs:
                corner_pos = self.compute_corner_position(cluster_info, pair)
                
                size1 = cluster_info[pair[0]]['size']
                size2 = cluster_info[pair[1]]['size']
                confidence = (size1 + size2) / len(significant_points)
                
                corners.append({
                    'position': corner_pos,
                    'cluster_labels': list(pair),
                    'confidence': confidence,
                    'type': 'paired_clusters',
                    'cluster_sizes': [size1, size2]
                })
        
        # 如果没有找到配对簇，使用加权质心
        if len(corners) == 0:
            corners.append({
                'position': weighted_center,
                'cluster_labels': list(cluster_info.keys()) if cluster_info else [0],
                'confidence': 1.0,
                'type': 'weighted_centroid'
            })
        
        # 按置信度排序
        corners.sort(key=lambda x: x['confidence'], reverse=True)
        
        return corners
    
    def visualize_clustering(self,
                              significant_points: np.ndarray,
                              image: np.ndarray = None,
                              corner_gt: Tuple[int, int] = None,
                              save_path: str = None):
        """
        可视化聚类结果
        
        Parameters:
        -----------
        significant_points : np.ndarray
            显著响应点
        image : np.ndarray
            背景图像（可选）
        corner_gt : tuple
            角点真值位置
        save_path : str
            保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 执行聚类
        labels, cluster_info = self.hierarchical_clustering(significant_points)
        corners = self.localize_corners(significant_points)
        
        # 原始响应点
        if image is not None:
            axes[0].imshow(image, cmap='gray', alpha=0.5)
        axes[0].scatter(significant_points[:, 1], significant_points[:, 0],
                       c='blue', s=20, alpha=0.7)
        if corner_gt:
            axes[0].plot(corner_gt[0], corner_gt[1], 'r+', markersize=15, 
                        markeredgewidth=3, label='Ground Truth')
        axes[0].set_title('Significant Response Points')
        axes[0].legend()
        axes[0].axis('off')
        
        # 聚类结果
        if image is not None:
            axes[1].imshow(image, cmap='gray', alpha=0.5)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))
        for i, label in enumerate(np.unique(labels)):
            mask = labels == label
            cluster_points = significant_points[mask]
            axes[1].scatter(cluster_points[:, 1], cluster_points[:, 0],
                           c=[colors[i % len(colors)]], s=30, alpha=0.7,
                           label=f'Cluster {label}')
            
            # 绘制质心
            if label in cluster_info:
                centroid = cluster_info[label]['centroid']
                axes[1].plot(centroid[1], centroid[0], 'o', 
                            color=colors[i % len(colors)], markersize=10,
                            markeredgecolor='black', markeredgewidth=2)
        
        axes[1].set_title(f'Hierarchical Clustering (N_clusters={len(cluster_info)})')
        axes[1].axis('off')
        
        # 角点定位结果
        if image is not None:
            axes[2].imshow(image, cmap='gray', alpha=0.5)
        
        axes[2].scatter(significant_points[:, 1], significant_points[:, 0],
                       c='lightgray', s=10, alpha=0.5)
        
        for corner in corners:
            pos = corner['position']
            conf = corner['confidence']
            marker = 's' if corner['type'] == 'paired_clusters' else '^'
            axes[2].plot(pos[1], pos[0], marker, color='lime', markersize=15,
                        markeredgecolor='black', markeredgewidth=2,
                        label=f"Corner (conf={conf:.2f})")
        
        if corner_gt:
            axes[2].plot(corner_gt[0], corner_gt[1], 'r+', markersize=15,
                        markeredgewidth=3, label='Ground Truth')
        
        axes[2].set_title('Detected Corners')
        axes[2].legend(loc='upper right')
        axes[2].axis('off')
        
        plt.suptitle('Corner Localization via Hierarchical Clustering', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return corners


class LocalizationParameters:
    """角点定位参数配置类（对应论文表2）"""
    
    # 默认参数
    DISTANCE_THRESHOLD_RANGE = (10, 20)   # T_d: 聚类距离阈值
    MIN_CLUSTER_SIZE_RANGE = (3, 5)       # N_min: 最小簇点数
    MAX_PAIRING_DISTANCE_RANGE = (30, 50) # D_max: 最大配对距离
    
    @classmethod
    def get_default_params(cls) -> dict:
        """获取默认参数配置"""
        return {
            'distance_threshold': 15.0,
            'min_cluster_size': 3,
            'max_pairing_distance': 40.0,
            'linkage_method': 'average'
        }


def compute_localization_error(detected_pos: np.ndarray, 
                                ground_truth: Tuple[int, int]) -> float:
    """
    计算定位误差
    
    Parameters:
    -----------
    detected_pos : np.ndarray
        检测到的角点位置 [y, x]
    ground_truth : tuple
        真值位置 (x, y)
        
    Returns:
    --------
    float
        欧氏距离误差（像素）
    """
    gt_array = np.array([ground_truth[1], ground_truth[0]])  # 转换为 [y, x]
    error = np.linalg.norm(detected_pos - gt_array)
    return error


def demo_corner_localization():
    """演示角点定位"""
    from module1_synthetic_data import SyntheticCornerGenerator
    from module2_simple_cell import SimpleCellModel
    from module3_end_stopped_cell import EndStoppedCellModel
    
    # 创建各模块
    generator = SyntheticCornerGenerator(image_size=256)
    simple_cell = SimpleCellModel(wavelength=6.0, n_orientations=12)
    end_stopped_cell = EndStoppedCellModel(sigma_s=4.0, response_threshold_percentile=80)
    corner_localizer = CornerLocalizer(
        distance_threshold=15.0,
        min_cluster_size=3,
        max_pairing_distance=40.0
    )
    
    # 测试不同角度
    test_angles = [30, 45, 60, 90, 120, 150]
    results = []
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, angle in enumerate(test_angles):
        img, corner_pos = generator.generate_single_test_image(angle, add_noise=False)
        
        # 完整处理流程
        edge_mag, edge_ori = simple_cell.compute_edge_response(img)
        edge_points = simple_cell.extract_edge_points(edge_mag, threshold_percentile=75)
        sig_points, responses = end_stopped_cell.dynamic_construct(img, edge_points, edge_ori)
        corners = corner_localizer.localize_corners(sig_points, responses)
        
        # 计算误差
        if len(corners) > 0:
            detected_pos = corners[0]['position']
            error = compute_localization_error(detected_pos, corner_pos)
        else:
            detected_pos = None
            error = float('inf')
        
        results.append({
            'angle': angle,
            'error': error,
            'detected_pos': detected_pos,
            'ground_truth': corner_pos,
            'n_clusters': len(corners)
        })
        
        # 可视化
        axes[idx].imshow(img, cmap='gray')
        axes[idx].scatter(sig_points[:, 1], sig_points[:, 0], 
                         c='yellow', s=10, alpha=0.5, label='Response Points')
        axes[idx].plot(corner_pos[0], corner_pos[1], 'r+', 
                      markersize=15, markeredgewidth=3, label='Ground Truth')
        
        if detected_pos is not None:
            axes[idx].plot(detected_pos[1], detected_pos[0], 'gs',
                          markersize=12, markeredgewidth=2, 
                          markerfacecolor='none', label=f'Detected (err={error:.2f}px)')
        
        axes[idx].set_title(f'Angle: {angle}°, Error: {error:.2f} px')
        axes[idx].legend(loc='upper right', fontsize=8)
        axes[idx].axis('off')
    
    plt.suptitle('Corner Localization Results at Different Angles', fontsize=14)
    plt.tight_layout()
    plt.savefig('./corner_localization_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印结果汇总
    print("\n" + "="*60)
    print("Corner Localization Results Summary")
    print("="*60)
    print(f"{'Angle':>8} | {'Error (px)':>12} | {'Status':>10}")
    print("-"*60)
    for r in results:
        status = 'Success' if r['error'] < 5.0 else 'Warning' if r['error'] < 10.0 else 'Failed'
        print(f"{r['angle']:>8}° | {r['error']:>12.2f} | {status:>10}")
    print("="*60)
    print(f"Mean Error: {np.mean([r['error'] for r in results if r['error'] < float('inf')]):.2f} px")
    
    return results


if __name__ == "__main__":
    demo_corner_localization()
