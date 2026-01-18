"""
Module 2: Simple Cell Edge Filtering Based on Gabor Filters
基于简单细胞与端抑制细胞协同处理的仿生弯角检测方法 - 简单细胞边缘滤波模块

This module implements Gabor filter bank to simulate simple cell responses
in V1 visual cortex for edge detection and orientation extraction.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class SimpleCellModel:
    """
    简单细胞模型 - 基于Gabor滤波器组
    
    模拟V1区简单细胞的方向选择性特性，提取图像中的边缘位置和方向信息
    """
    
    def __init__(self,
                 wavelength: float = 6.0,
                 sigma: float = None,
                 gamma: float = 0.5,
                 psi: float = 0.0,
                 n_orientations: int = 12,
                 kernel_size: int = None):
        """
        初始化简单细胞模型
        
        Parameters:
        -----------
        wavelength : float
            Gabor滤波器波长 λ (像素)
        sigma : float
            高斯包络标准差，默认为 0.56 * wavelength
        gamma : float
            空间纵横比 γ
        psi : float
            相位偏移 ψ
        n_orientations : int
            方向通道数 N_θ
        kernel_size : int
            滤波器核大小，默认自动计算
        """
        self.wavelength = wavelength
        self.sigma = sigma if sigma else 0.56 * wavelength
        self.gamma = gamma
        self.psi = psi
        self.n_orientations = n_orientations
        
        # 自动计算核大小
        if kernel_size is None:
            self.kernel_size = int(6 * self.sigma)
            if self.kernel_size % 2 == 0:
                self.kernel_size += 1
        else:
            self.kernel_size = kernel_size
        
        # 生成方向角度列表
        self.orientations = np.array([k * np.pi / n_orientations for k in range(n_orientations)])
        
        # 预计算Gabor滤波器组
        self.gabor_filters = self._build_gabor_bank()
        
    def _build_gabor_bank(self) -> List[np.ndarray]:
        """
        构建Gabor滤波器组
        
        Returns:
        --------
        list
            不同方向的Gabor滤波器列表
        """
        filters = []
        
        for theta in self.orientations:
            kernel = self._create_gabor_kernel(theta)
            filters.append(kernel)
        
        return filters
    
    def _create_gabor_kernel(self, theta: float) -> np.ndarray:
        """
        创建单个Gabor滤波器核
        
        G(x,y) = exp(-(x'^2 + γ²y'^2)/(2σ²)) * cos(2π*x'/λ + ψ)
        其中: x' = x*cos(θ) + y*sin(θ)
              y' = -x*sin(θ) + y*cos(θ)
        
        Parameters:
        -----------
        theta : float
            方向角度（弧度）
            
        Returns:
        --------
        np.ndarray
            Gabor滤波器核
        """
        half_size = self.kernel_size // 2
        x, y = np.meshgrid(
            np.arange(-half_size, half_size + 1),
            np.arange(-half_size, half_size + 1)
        )
        
        # 坐标旋转
        x_prime = x * np.cos(theta) + y * np.sin(theta)
        y_prime = -x * np.sin(theta) + y * np.cos(theta)
        
        # Gabor函数
        gaussian = np.exp(-(x_prime**2 + self.gamma**2 * y_prime**2) / (2 * self.sigma**2))
        sinusoid = np.cos(2 * np.pi * x_prime / self.wavelength + self.psi)
        
        kernel = gaussian * sinusoid
        
        # 归一化
        kernel = kernel - kernel.mean()
        kernel = kernel / np.sqrt(np.sum(kernel**2))
        
        return kernel.astype(np.float32)
    
    def compute_edge_response(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算边缘响应
        
        对输入图像进行Gabor滤波，返回综合边缘强度图和主导方向图
        
        Parameters:
        -----------
        image : np.ndarray
            输入灰度图像
            
        Returns:
        --------
        tuple
            (edge_magnitude, edge_orientation)
            - edge_magnitude: 边缘强度图 E(x,y)
            - edge_orientation: 边缘方向图 Θ(x,y)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = image.astype(np.float32)
        
        # 存储所有方向的响应
        responses = np.zeros((self.n_orientations, image.shape[0], image.shape[1]), dtype=np.float32)
        
        for k, kernel in enumerate(self.gabor_filters):
            response = cv2.filter2D(image, cv2.CV_32F, kernel)
            responses[k] = np.abs(response)
        
        # 计算综合边缘强度（各方向响应的最大值）
        edge_magnitude = np.max(responses, axis=0)
        
        # 计算主导方向（响应最大的通道索引）
        max_indices = np.argmax(responses, axis=0)
        edge_orientation = self.orientations[max_indices]
        
        return edge_magnitude, edge_orientation
    
    def extract_edge_points(self, 
                           edge_magnitude: np.ndarray, 
                           threshold_percentile: float = 75) -> np.ndarray:
        """
        提取候选边缘点
        
        通过阈值筛选确定候选边缘点集合 E
        
        Parameters:
        -----------
        edge_magnitude : np.ndarray
            边缘强度图
        threshold_percentile : float
            阈值百分位数 (70%-85%)
            
        Returns:
        --------
        np.ndarray
            边缘点坐标 (N, 2)，每行为 (y, x)
        """
        threshold = np.percentile(edge_magnitude, threshold_percentile)
        edge_mask = edge_magnitude > threshold
        
        # 非极大值抑制
        edge_nms = self._non_maximum_suppression(edge_magnitude, edge_mask)
        
        # 获取边缘点坐标
        edge_points = np.argwhere(edge_nms)
        
        return edge_points
    
    def _non_maximum_suppression(self, 
                                  magnitude: np.ndarray, 
                                  mask: np.ndarray) -> np.ndarray:
        """
        非极大值抑制
        
        Parameters:
        -----------
        magnitude : np.ndarray
            边缘强度图
        mask : np.ndarray
            边缘掩码
            
        Returns:
        --------
        np.ndarray
            抑制后的边缘图
        """
        result = np.zeros_like(magnitude, dtype=np.uint8)
        
        # 使用形态学操作进行局部极大值检测
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(magnitude, kernel)
        
        local_max = (magnitude >= dilated) & mask
        result[local_max] = 255
        
        return result
    
    def visualize_gabor_bank(self, save_path: str = None):
        """
        可视化Gabor滤波器组
        
        Parameters:
        -----------
        save_path : str
            保存路径（可选）
        """
        n_cols = min(6, self.n_orientations)
        n_rows = (self.n_orientations + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes.flatten() if self.n_orientations > 1 else [axes]
        
        for k, (kernel, ax) in enumerate(zip(self.gabor_filters, axes)):
            ax.imshow(kernel, cmap='RdBu_r')
            ax.set_title(f'θ = {np.degrees(self.orientations[k]):.1f}°')
            ax.axis('off')
        
        # 隐藏多余的子图
        for ax in axes[len(self.gabor_filters):]:
            ax.axis('off')
        
        plt.suptitle('Gabor Filter Bank (Simple Cell Model)', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_edge_detection(self, 
                                  image: np.ndarray, 
                                  threshold_percentile: float = 75,
                                  save_path: str = None):
        """
        可视化边缘检测结果
        
        Parameters:
        -----------
        image : np.ndarray
            输入图像
        threshold_percentile : float
            阈值百分位数
        save_path : str
            保存路径（可选）
        """
        edge_mag, edge_ori = self.compute_edge_response(image)
        edge_points = self.extract_edge_points(edge_mag, threshold_percentile)
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # 原图
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')
        
        # 边缘强度图
        axes[0, 1].imshow(edge_mag, cmap='hot')
        axes[0, 1].set_title('Edge Magnitude E(x,y)')
        axes[0, 1].axis('off')
        
        # 边缘方向图
        axes[1, 0].imshow(edge_ori, cmap='hsv')
        axes[1, 0].set_title('Edge Orientation Θ(x,y)')
        axes[1, 0].axis('off')
        
        # 候选边缘点
        edge_overlay = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()
        for point in edge_points:
            cv2.circle(edge_overlay, (point[1], point[0]), 1, (0, 255, 0), -1)
        axes[1, 1].imshow(edge_overlay)
        axes[1, 1].set_title(f'Edge Points (N={len(edge_points)})')
        axes[1, 1].axis('off')
        
        plt.suptitle('Simple Cell Edge Filtering Results', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return edge_mag, edge_ori, edge_points


class GaborFilterParameters:
    """Gabor滤波器参数配置类（对应论文表1）"""
    
    # 默认参数
    WAVELENGTH_RANGE = (4, 8)      # λ: Gabor波长（像素）
    SIGMA_FACTOR = 0.56            # σ = 0.56 * λ
    GAMMA = 0.5                    # γ: 空间纵横比
    PSI_OPTIONS = [0, np.pi/2]     # ψ: 相位偏移
    N_ORIENTATIONS_RANGE = (8, 16) # N_θ: 方向通道数
    THRESHOLD_RANGE = (70, 85)     # T_e: 边缘阈值百分位
    
    @classmethod
    def get_default_params(cls) -> dict:
        """获取默认参数配置"""
        return {
            'wavelength': 6.0,
            'sigma': 0.56 * 6.0,
            'gamma': 0.5,
            'psi': 0.0,
            'n_orientations': 12,
            'threshold_percentile': 75
        }


def demo_simple_cell():
    """演示简单细胞边缘检测"""
    from module1_synthetic_data import SyntheticCornerGenerator
    
    # 生成测试图像
    generator = SyntheticCornerGenerator(image_size=256)
    test_angles = [30, 60, 90, 120]
    
    # 创建简单细胞模型
    simple_cell = SimpleCellModel(
        wavelength=6.0,
        n_orientations=12
    )
    
    # 可视化Gabor滤波器组
    simple_cell.visualize_gabor_bank(save_path='./gabor_filter_bank.png')
    
    # 对不同角度进行边缘检测
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    
    for row, angle in enumerate(test_angles):
        img, corner_pos = generator.generate_single_test_image(angle, add_noise=False)
        edge_mag, edge_ori = simple_cell.compute_edge_response(img)
        edge_points = simple_cell.extract_edge_points(edge_mag, threshold_percentile=75)
        
        # 原图
        axes[row, 0].imshow(img, cmap='gray')
        axes[row, 0].plot(corner_pos[0], corner_pos[1], 'r+', markersize=10)
        axes[row, 0].set_title(f'Input ({angle}°)')
        axes[row, 0].axis('off')
        
        # 边缘强度
        axes[row, 1].imshow(edge_mag, cmap='hot')
        axes[row, 1].set_title('Edge Magnitude')
        axes[row, 1].axis('off')
        
        # 边缘方向
        axes[row, 2].imshow(edge_ori, cmap='hsv')
        axes[row, 2].set_title('Edge Orientation')
        axes[row, 2].axis('off')
        
        # 边缘点
        edge_img = np.zeros_like(img)
        for pt in edge_points:
            edge_img[pt[0], pt[1]] = 255
        axes[row, 3].imshow(edge_img, cmap='gray')
        axes[row, 3].set_title(f'Edge Points (N={len(edge_points)})')
        axes[row, 3].axis('off')
    
    plt.suptitle('Simple Cell Edge Detection at Different Angles', fontsize=14)
    plt.tight_layout()
    plt.savefig('./simple_cell_edge_detection.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    demo_simple_cell()
