"""
Module 3: End-Stopped Cell Dynamic Construction
基于简单细胞与端抑制细胞协同处理的仿生弯角检测方法 - 端抑制细胞动态构建模块

This module implements unilateral LoG (Laplacian of Gaussian) operators to simulate
end-stopped cell responses for corner and endpoint detection.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from scipy import ndimage


class EndStoppedCellModel:
    """
    端抑制细胞模型 - 基于单侧LoG算子
    
    模拟视觉皮层端抑制细胞的响应特性，检测边缘端点和角点结构
    """
    
    def __init__(self,
                 sigma_s: float = 4.0,
                 mask_width: float = 2.0,
                 neighborhood_size: int = 15,
                 response_threshold_percentile: float = 85):
        """
        初始化端抑制细胞模型
        
        Parameters:
        -----------
        sigma_s : float
            LoG算子尺度参数 σ_s
        mask_width : float
            方向掩模宽度参数 α
        neighborhood_size : int
            局部邻域大小 K
        response_threshold_percentile : float
            响应阈值百分位数
        """
        self.sigma_s = sigma_s
        self.mask_width = mask_width
        self.neighborhood_size = neighborhood_size
        self.response_threshold_percentile = response_threshold_percentile
        
        # 预计算LoG核
        self.log_kernel = self._create_log_kernel()
        
    def _create_log_kernel(self) -> np.ndarray:
        """
        创建LoG（高斯拉普拉斯）算子核
        
        LoG(x,y;σ) = -1/(πσ⁴) * (1 - (x²+y²)/(2σ²)) * exp(-(x²+y²)/(2σ²))
        
        Returns:
        --------
        np.ndarray
            LoG算子核
        """
        # 核大小
        size = int(6 * self.sigma_s)
        if size % 2 == 0:
            size += 1
        half_size = size // 2
        
        x, y = np.meshgrid(
            np.arange(-half_size, half_size + 1),
            np.arange(-half_size, half_size + 1)
        )
        
        r_squared = x**2 + y**2
        sigma_squared = self.sigma_s ** 2
        
        # LoG公式
        term1 = -1 / (np.pi * sigma_squared ** 2)
        term2 = 1 - r_squared / (2 * sigma_squared)
        term3 = np.exp(-r_squared / (2 * sigma_squared))
        
        log_kernel = term1 * term2 * term3
        
        # 归一化使和为零
        log_kernel = log_kernel - log_kernel.mean()
        
        return log_kernel.astype(np.float32)
    
    def _create_directional_mask(self, 
                                  theta: float, 
                                  size: int,
                                  side: str = 'both') -> np.ndarray:
        """
        创建方向掩模
        
        M(x,y;θ,α) = 1 if |(x-x₀)sin(θ) - (y-y₀)cos(θ)| < α
                   = 0 otherwise
        
        Parameters:
        -----------
        theta : float
            边缘方向（弧度）
        size : int
            掩模大小
        side : str
            掩模侧向 'left', 'right', 或 'both'
            
        Returns:
        --------
        np.ndarray
            方向掩模
        """
        half_size = size // 2
        x, y = np.meshgrid(
            np.arange(-half_size, half_size + 1),
            np.arange(-half_size, half_size + 1)
        )
        
        # 计算到边缘方向线的距离
        # 边缘方向为theta，垂直方向为theta + π/2
        perpendicular_dist = np.abs(x * np.sin(theta) - y * np.cos(theta))
        
        # 基础掩模
        mask = (perpendicular_dist < self.mask_width * self.sigma_s).astype(np.float32)
        
        # 单侧掩模
        if side == 'left':
            # 保留边缘左侧
            parallel_dist = x * np.cos(theta) + y * np.sin(theta)
            mask = mask * (parallel_dist < 0).astype(np.float32)
        elif side == 'right':
            # 保留边缘右侧
            parallel_dist = x * np.cos(theta) + y * np.sin(theta)
            mask = mask * (parallel_dist >= 0).astype(np.float32)
        
        return mask
    
    def compute_response_at_point(self,
                                   image: np.ndarray,
                                   point: Tuple[int, int],
                                   orientation: float,
                                   side: str = 'right') -> float:
        """
        计算单个点的端抑制细胞响应
        
        S(x₀,y₀) = Σ I(x,y) · LoG(x-x₀,y-y₀) · M(x,y;θ,α)
        
        Parameters:
        -----------
        image : np.ndarray
            输入图像
        point : tuple
            计算点坐标 (y, x)
        orientation : float
            该点的边缘方向（弧度）
        side : str
            掩模侧向
            
        Returns:
        --------
        float
            端抑制细胞响应值
        """
        y0, x0 = point
        half_size = self.neighborhood_size // 2
        
        # 确保不超出图像边界
        y_start = max(0, y0 - half_size)
        y_end = min(image.shape[0], y0 + half_size + 1)
        x_start = max(0, x0 - half_size)
        x_end = min(image.shape[1], x0 + half_size + 1)
        
        # 提取局部区域
        local_region = image[y_start:y_end, x_start:x_end].astype(np.float32)
        
        # 计算LoG响应
        if local_region.shape[0] < 3 or local_region.shape[1] < 3:
            return 0.0
        
        # 调整LoG核大小以匹配局部区域
        log_response = cv2.filter2D(local_region, cv2.CV_32F, self.log_kernel)
        
        # 创建方向掩模
        mask = self._create_directional_mask(orientation, local_region.shape[0], side)
        
        # 调整掩模大小
        if mask.shape != log_response.shape:
            mask = cv2.resize(mask, (log_response.shape[1], log_response.shape[0]))
        
        # 计算加权响应
        response = np.sum(log_response * mask)
        
        return response
    
    def dynamic_construct(self,
                          image: np.ndarray,
                          edge_points: np.ndarray,
                          edge_orientations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        动态构建端抑制细胞
        
        仅在筛选出的边缘位置上构建单侧端抑制细胞
        
        Parameters:
        -----------
        image : np.ndarray
            输入图像
        edge_points : np.ndarray
            候选边缘点坐标 (N, 2)
        edge_orientations : np.ndarray
            边缘方向图
            
        Returns:
        --------
        tuple
            (significant_points, responses)
            - significant_points: 显著响应点坐标
            - responses: 对应的响应值
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = image.astype(np.float32)
        
        # 计算完整LoG响应图
        log_response = cv2.filter2D(image, cv2.CV_32F, self.log_kernel)
        log_response = np.abs(log_response)
        
        responses = []
        
        # 对每个边缘点获取LoG响应
        for point in edge_points:
            y, x = point
            if 0 <= y < log_response.shape[0] and 0 <= x < log_response.shape[1]:
                response = log_response[y, x]
                responses.append(response)
            else:
                responses.append(0)
        
        responses = np.array(responses)
        
        # 阈值筛选显著响应点
        if len(responses) > 0 and responses.max() > 0:
            threshold = np.percentile(responses, self.response_threshold_percentile)
            significant_mask = responses > threshold
            significant_points = edge_points[significant_mask]
            significant_responses = responses[significant_mask]
        else:
            significant_points = np.array([]).reshape(0, 2)
            significant_responses = np.array([])
        
        return significant_points, significant_responses
    
    def compute_full_response_map(self, 
                                   image: np.ndarray,
                                   edge_orientations: np.ndarray) -> np.ndarray:
        """
        计算完整的端抑制细胞响应图（用于可视化）
        
        Parameters:
        -----------
        image : np.ndarray
            输入图像
        edge_orientations : np.ndarray
            边缘方向图
            
        Returns:
        --------
        np.ndarray
            响应强度图
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = image.astype(np.float32)
        
        # 计算基础LoG响应
        log_response = cv2.filter2D(image, cv2.CV_32F, self.log_kernel)
        
        # 取绝对值
        response_map = np.abs(log_response)
        
        return response_map
    
    def visualize_log_kernel(self, save_path: str = None):
        """
        可视化LoG算子核
        
        Parameters:
        -----------
        save_path : str
            保存路径（可选）
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # 2D视图
        im = axes[0].imshow(self.log_kernel, cmap='RdBu_r')
        axes[0].set_title('LoG Kernel (2D)')
        axes[0].axis('off')
        plt.colorbar(im, ax=axes[0], fraction=0.046)
        
        # 3D视图
        from mpl_toolkits.mplot3d import Axes3D
        ax3d = fig.add_subplot(1, 3, 2, projection='3d')
        x = np.arange(self.log_kernel.shape[1])
        y = np.arange(self.log_kernel.shape[0])
        X, Y = np.meshgrid(x, y)
        ax3d.plot_surface(X, Y, self.log_kernel, cmap='RdBu_r')
        ax3d.set_title('LoG Kernel (3D)')
        axes[1].axis('off')
        
        # 剖面图
        center = self.log_kernel.shape[0] // 2
        profile = self.log_kernel[center, :]
        axes[2].plot(profile, 'b-', linewidth=2)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[2].set_title('LoG Kernel Profile')
        axes[2].set_xlabel('Position')
        axes[2].set_ylabel('Value')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Laplacian of Gaussian (σ = {self.sigma_s})', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_response(self,
                           image: np.ndarray,
                           edge_points: np.ndarray,
                           significant_points: np.ndarray,
                           responses: np.ndarray,
                           corner_gt: Tuple[int, int] = None,
                           save_path: str = None):
        """
        可视化端抑制细胞响应
        
        Parameters:
        -----------
        image : np.ndarray
            输入图像
        edge_points : np.ndarray
            所有边缘点
        significant_points : np.ndarray
            显著响应点
        responses : np.ndarray
            响应值
        corner_gt : tuple
            角点真值位置
        save_path : str
            保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图和边缘点
        axes[0].imshow(image, cmap='gray')
        if len(edge_points) > 0:
            axes[0].scatter(edge_points[:, 1], edge_points[:, 0], 
                           c='green', s=1, alpha=0.5, label='Edge Points')
        if corner_gt:
            axes[0].plot(corner_gt[0], corner_gt[1], 'r+', markersize=15, 
                        markeredgewidth=3, label='Ground Truth')
        axes[0].set_title(f'Edge Points (N={len(edge_points)})')
        axes[0].legend()
        axes[0].axis('off')
        
        # 显著响应点
        axes[1].imshow(image, cmap='gray')
        if len(significant_points) > 0:
            sc = axes[1].scatter(significant_points[:, 1], significant_points[:, 0],
                                c=responses, cmap='hot', s=20, alpha=0.8)
            plt.colorbar(sc, ax=axes[1], fraction=0.046, label='Response')
        if corner_gt:
            axes[1].plot(corner_gt[0], corner_gt[1], 'g+', markersize=15, 
                        markeredgewidth=3, label='Ground Truth')
        axes[1].set_title(f'Significant Response Points (N={len(significant_points)})')
        axes[1].axis('off')
        
        # 响应强度分布
        if len(responses) > 0:
            axes[2].hist(responses, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
            threshold = np.percentile(responses, self.response_threshold_percentile)
            axes[2].axvline(x=threshold, color='red', linestyle='--', 
                           label=f'Threshold ({self.response_threshold_percentile}%)')
            axes[2].set_xlabel('Response Value')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title('Response Distribution')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('End-Stopped Cell Response', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class EndStoppedCellParameters:
    """端抑制细胞参数配置类（对应论文表2）"""
    
    # 默认参数
    SIGMA_S_RANGE = (3, 6)           # σ_s: LoG尺度参数
    MASK_WIDTH_RANGE = (1.5, 3.0)    # α: 掩模宽度
    NEIGHBORHOOD_SIZE = 15           # K: 邻域大小
    THRESHOLD_RANGE = (80, 90)       # T_s: 响应阈值百分位
    
    @classmethod
    def get_default_params(cls) -> dict:
        """获取默认参数配置"""
        return {
            'sigma_s': 4.0,
            'mask_width': 2.0,
            'neighborhood_size': 15,
            'response_threshold_percentile': 85
        }


def demo_end_stopped_cell():
    """演示端抑制细胞响应"""
    from module1_synthetic_data import SyntheticCornerGenerator
    from module2_simple_cell import SimpleCellModel
    
    # 生成测试图像
    generator = SyntheticCornerGenerator(image_size=256)
    
    # 创建简单细胞和端抑制细胞模型
    simple_cell = SimpleCellModel(wavelength=6.0, n_orientations=12)
    end_stopped_cell = EndStoppedCellModel(sigma_s=4.0, response_threshold_percentile=80)
    
    # 可视化LoG核
    end_stopped_cell.visualize_log_kernel(save_path='./log_kernel.png')
    
    # 对不同角度进行处理
    test_angles = [30, 60, 90, 120]
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    for row, angle in enumerate(test_angles):
        img, corner_pos = generator.generate_single_test_image(angle, add_noise=False)
        
        # 简单细胞边缘检测
        edge_mag, edge_ori = simple_cell.compute_edge_response(img)
        edge_points = simple_cell.extract_edge_points(edge_mag, threshold_percentile=75)
        
        # 端抑制细胞动态构建
        sig_points, responses = end_stopped_cell.dynamic_construct(img, edge_points, edge_ori)
        
        # 可视化
        axes[row, 0].imshow(img, cmap='gray')
        axes[row, 0].plot(corner_pos[0], corner_pos[1], 'r+', markersize=12, markeredgewidth=2)
        axes[row, 0].set_title(f'Input ({angle}°)')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(edge_mag, cmap='hot')
        axes[row, 1].set_title('Edge Magnitude')
        axes[row, 1].axis('off')
        
        # 边缘点
        edge_img = np.zeros_like(img)
        for pt in edge_points:
            edge_img[pt[0], pt[1]] = 255
        axes[row, 2].imshow(edge_img, cmap='gray')
        axes[row, 2].set_title(f'Edge Points (N={len(edge_points)})')
        axes[row, 2].axis('off')
        
        # 显著响应点
        axes[row, 3].imshow(img, cmap='gray')
        if len(sig_points) > 0:
            axes[row, 3].scatter(sig_points[:, 1], sig_points[:, 0],
                                c=responses, cmap='hot', s=30, alpha=0.8)
        axes[row, 3].plot(corner_pos[0], corner_pos[1], 'g+', markersize=12, markeredgewidth=2)
        axes[row, 3].set_title(f'End-Stopped Response (N={len(sig_points)})')
        axes[row, 3].axis('off')
    
    plt.suptitle('End-Stopped Cell Dynamic Construction', fontsize=14)
    plt.tight_layout()
    plt.savefig('./end_stopped_cell_response.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    demo_end_stopped_cell()
