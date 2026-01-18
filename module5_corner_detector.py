"""
Module 5: Bionic Corner Detection Pipeline
基于简单细胞与端抑制细胞协同处理的仿生弯角检测方法 - 完整检测流程模块

This module integrates all components to provide a complete corner detection
pipeline based on cooperative processing of simple cells and end-stopped cells.
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
import time

from module2_simple_cell import SimpleCellModel, GaborFilterParameters
from module3_end_stopped_cell import EndStoppedCellModel, EndStoppedCellParameters
from module4_corner_localization import CornerLocalizer, LocalizationParameters


class BionicCornerDetector:
    """
    仿生角点检测器
    
    基于简单细胞与端抑制细胞协同处理的两阶段角点检测方法
    """
    
    def __init__(self,
                 # 简单细胞参数
                 gabor_wavelength: float = 6.0,
                 gabor_n_orientations: int = 12,
                 edge_threshold_percentile: float = 75,
                 # 端抑制细胞参数
                 log_sigma: float = 4.0,
                 response_threshold_percentile: float = 80,
                 # 角点定位参数
                 cluster_distance_threshold: float = 15.0,
                 min_cluster_size: int = 3,
                 max_pairing_distance: float = 40.0):
        """
        初始化仿生角点检测器
        
        Parameters:
        -----------
        gabor_wavelength : float
            Gabor滤波器波长
        gabor_n_orientations : int
            Gabor方向通道数
        edge_threshold_percentile : float
            边缘阈值百分位
        log_sigma : float
            LoG尺度参数
        response_threshold_percentile : float
            端抑制细胞响应阈值百分位
        cluster_distance_threshold : float
            聚类距离阈值
        min_cluster_size : int
            最小簇点数
        max_pairing_distance : float
            最大配对距离
        """
        # 创建各阶段模型
        self.simple_cell = SimpleCellModel(
            wavelength=gabor_wavelength,
            n_orientations=gabor_n_orientations
        )
        
        self.end_stopped_cell = EndStoppedCellModel(
            sigma_s=log_sigma,
            response_threshold_percentile=response_threshold_percentile
        )
        
        self.corner_localizer = CornerLocalizer(
            distance_threshold=cluster_distance_threshold,
            min_cluster_size=min_cluster_size,
            max_pairing_distance=max_pairing_distance
        )
        
        # 保存参数
        self.params = {
            'gabor_wavelength': gabor_wavelength,
            'gabor_n_orientations': gabor_n_orientations,
            'edge_threshold_percentile': edge_threshold_percentile,
            'log_sigma': log_sigma,
            'response_threshold_percentile': response_threshold_percentile,
            'cluster_distance_threshold': cluster_distance_threshold,
            'min_cluster_size': min_cluster_size,
            'max_pairing_distance': max_pairing_distance
        }
        
        self.edge_threshold_percentile = edge_threshold_percentile
    
    def detect(self, 
               image: np.ndarray, 
               return_intermediate: bool = False) -> Dict:
        """
        执行角点检测
        
        Parameters:
        -----------
        image : np.ndarray
            输入图像（灰度或彩色）
        return_intermediate : bool
            是否返回中间结果
            
        Returns:
        --------
        dict
            检测结果，包含角点位置和相关信息
        """
        start_time = time.time()
        
        # 预处理
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 阶段1：简单细胞边缘滤波
        t1 = time.time()
        edge_magnitude, edge_orientation = self.simple_cell.compute_edge_response(gray)
        edge_points = self.simple_cell.extract_edge_points(
            edge_magnitude, 
            self.edge_threshold_percentile
        )
        stage1_time = time.time() - t1
        
        # 阶段2：端抑制细胞动态构建
        t2 = time.time()
        significant_points, responses = self.end_stopped_cell.dynamic_construct(
            gray, 
            edge_points, 
            edge_orientation
        )
        stage2_time = time.time() - t2
        
        # 阶段3：角点定位
        t3 = time.time()
        corners = self.corner_localizer.localize_corners(significant_points, responses)
        stage3_time = time.time() - t3
        
        total_time = time.time() - start_time
        
        # 构建结果
        result = {
            'corners': corners,
            'n_corners': len(corners),
            'timing': {
                'total': total_time * 1000,  # 转换为毫秒
                'edge_filtering': stage1_time * 1000,
                'end_stopped_construction': stage2_time * 1000,
                'corner_localization': stage3_time * 1000
            },
            'statistics': {
                'n_edge_points': len(edge_points),
                'n_significant_points': len(significant_points)
            }
        }
        
        if return_intermediate:
            result['intermediate'] = {
                'edge_magnitude': edge_magnitude,
                'edge_orientation': edge_orientation,
                'edge_points': edge_points,
                'significant_points': significant_points,
                'responses': responses
            }
        
        return result
    
    def detect_and_visualize(self,
                              image: np.ndarray,
                              ground_truth: Tuple[int, int] = None,
                              save_path: str = None) -> Dict:
        """
        检测角点并可视化完整流程
        
        Parameters:
        -----------
        image : np.ndarray
            输入图像
        ground_truth : tuple
            角点真值位置 (x, y)
        save_path : str
            保存路径
            
        Returns:
        --------
        dict
            检测结果
        """
        result = self.detect(image, return_intermediate=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原图
        if len(image.shape) == 3:
            axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            axes[0, 0].imshow(image, cmap='gray')
        if ground_truth:
            axes[0, 0].plot(ground_truth[0], ground_truth[1], 'r+', 
                           markersize=15, markeredgewidth=3, label='Ground Truth')
        axes[0, 0].set_title('Input Image')
        axes[0, 0].legend()
        axes[0, 0].axis('off')
        
        # 边缘强度图
        axes[0, 1].imshow(result['intermediate']['edge_magnitude'], cmap='hot')
        axes[0, 1].set_title('Edge Magnitude (Simple Cell)')
        axes[0, 1].axis('off')
        
        # 边缘方向图
        axes[0, 2].imshow(result['intermediate']['edge_orientation'], cmap='hsv')
        axes[0, 2].set_title('Edge Orientation')
        axes[0, 2].axis('off')
        
        # 边缘点
        edge_img = np.zeros_like(image if len(image.shape) == 2 else image[:,:,0])
        for pt in result['intermediate']['edge_points']:
            edge_img[pt[0], pt[1]] = 255
        axes[1, 0].imshow(edge_img, cmap='gray')
        axes[1, 0].set_title(f"Edge Points (N={result['statistics']['n_edge_points']})")
        axes[1, 0].axis('off')
        
        # 端抑制细胞响应
        if len(image.shape) == 3:
            axes[1, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), alpha=0.5)
        else:
            axes[1, 1].imshow(image, cmap='gray', alpha=0.5)
        
        sig_pts = result['intermediate']['significant_points']
        if len(sig_pts) > 0:
            axes[1, 1].scatter(sig_pts[:, 1], sig_pts[:, 0],
                              c=result['intermediate']['responses'],
                              cmap='hot', s=20, alpha=0.8)
        axes[1, 1].set_title(f"End-Stopped Response (N={result['statistics']['n_significant_points']})")
        axes[1, 1].axis('off')
        
        # 最终检测结果
        if len(image.shape) == 3:
            axes[1, 2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            axes[1, 2].imshow(image, cmap='gray')
        
        for i, corner in enumerate(result['corners']):
            pos = corner['position']
            conf = corner['confidence']
            color = 'lime' if corner['type'] == 'paired_clusters' else 'yellow'
            axes[1, 2].plot(pos[1], pos[0], 's', color=color, markersize=12,
                           markeredgecolor='black', markeredgewidth=2)
            axes[1, 2].annotate(f'{conf:.2f}', (pos[1]+5, pos[0]-5), 
                               color='white', fontsize=8)
        
        if ground_truth:
            axes[1, 2].plot(ground_truth[0], ground_truth[1], 'r+',
                           markersize=15, markeredgewidth=3, label='Ground Truth')
            
            # 计算误差
            if len(result['corners']) > 0:
                detected = result['corners'][0]['position']
                error = np.sqrt((detected[1] - ground_truth[0])**2 + 
                               (detected[0] - ground_truth[1])**2)
                axes[1, 2].set_title(f"Detected Corners (Error: {error:.2f} px)")
            else:
                axes[1, 2].set_title("No Corners Detected")
        else:
            axes[1, 2].set_title(f"Detected Corners (N={result['n_corners']})")
        
        axes[1, 2].legend()
        axes[1, 2].axis('off')
        
        plt.suptitle(f"Bionic Corner Detection Pipeline (Total: {result['timing']['total']:.1f} ms)", 
                    fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return result
    
    def get_parameters(self) -> Dict:
        """获取当前参数配置"""
        return self.params.copy()
    
    def set_parameters(self, **kwargs):
        """更新参数配置"""
        # 重新初始化模型
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
        
        self.__init__(**self.params)


class BaselineCornerDetector:
    """
    基线方法：仅使用端抑制细胞的传统检测方法
    
    用于对比实验
    """
    
    def __init__(self, 
                 log_sigma: float = 4.0,
                 threshold_percentile: float = 90):
        """
        初始化基线检测器
        
        Parameters:
        -----------
        log_sigma : float
            LoG尺度参数
        threshold_percentile : float
            响应阈值百分位
        """
        self.log_sigma = log_sigma
        self.threshold_percentile = threshold_percentile
        self.log_kernel = self._create_log_kernel()
    
    def _create_log_kernel(self) -> np.ndarray:
        """创建LoG核"""
        size = int(6 * self.log_sigma)
        if size % 2 == 0:
            size += 1
        half_size = size // 2
        
        x, y = np.meshgrid(
            np.arange(-half_size, half_size + 1),
            np.arange(-half_size, half_size + 1)
        )
        
        r_squared = x**2 + y**2
        sigma_squared = self.log_sigma ** 2
        
        term1 = -1 / (np.pi * sigma_squared ** 2)
        term2 = 1 - r_squared / (2 * sigma_squared)
        term3 = np.exp(-r_squared / (2 * sigma_squared))
        
        log_kernel = term1 * term2 * term3
        log_kernel = log_kernel - log_kernel.mean()
        
        return log_kernel.astype(np.float32)
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        执行角点检测
        
        Parameters:
        -----------
        image : np.ndarray
            输入图像
            
        Returns:
        --------
        dict
            检测结果
        """
        start_time = time.time()
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        gray = gray.astype(np.float32)
        
        # 全局LoG响应
        response = cv2.filter2D(gray, cv2.CV_32F, self.log_kernel)
        response = np.abs(response)
        
        # 阈值筛选
        threshold = np.percentile(response, self.threshold_percentile)
        significant_mask = response > threshold
        
        # 非极大值抑制
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(response, kernel)
        local_max = (response >= dilated) & significant_mask
        
        # 获取响应点
        response_points = np.argwhere(local_max)
        
        # 计算重心作为角点位置
        if len(response_points) > 0:
            # 使用响应值加权的重心
            weights = response[local_max]
            weighted_y = np.sum(response_points[:, 0] * weights) / np.sum(weights)
            weighted_x = np.sum(response_points[:, 1] * weights) / np.sum(weights)
            corner_pos = np.array([weighted_y, weighted_x])
        else:
            corner_pos = None
        
        total_time = time.time() - start_time
        
        result = {
            'corners': [{'position': corner_pos, 'confidence': 1.0}] if corner_pos is not None else [],
            'n_corners': 1 if corner_pos is not None else 0,
            'timing': {'total': total_time * 1000},
            'response_map': response,
            'statistics': {
                'n_edge_points': len(response_points),
                'n_significant_points': len(response_points)
            }
        }
        
        return result


def compare_methods():
    """对比本方法与基线方法"""
    from module1_synthetic_data import SyntheticCornerGenerator
    
    generator = SyntheticCornerGenerator(image_size=256)
    bionic_detector = BionicCornerDetector()
    baseline_detector = BaselineCornerDetector()
    
    angles = [15, 30, 45, 60, 90, 120, 150, 165]
    
    results = {
        'bionic': {'errors': [], 'times': []},
        'baseline': {'errors': [], 'times': []}
    }
    
    for angle in angles:
        img, corner_pos = generator.generate_single_test_image(angle, add_noise=False)
        
        # 本方法
        bionic_result = bionic_detector.detect(img)
        if len(bionic_result['corners']) > 0:
            detected = bionic_result['corners'][0]['position']
            error = np.sqrt((detected[1] - corner_pos[0])**2 + 
                           (detected[0] - corner_pos[1])**2)
        else:
            error = float('inf')
        results['bionic']['errors'].append(error)
        results['bionic']['times'].append(bionic_result['timing']['total'])
        
        # 基线方法
        baseline_result = baseline_detector.detect(img)
        if len(baseline_result['corners']) > 0:
            detected = baseline_result['corners'][0]['position']
            error = np.sqrt((detected[1] - corner_pos[0])**2 + 
                           (detected[0] - corner_pos[1])**2)
        else:
            error = float('inf')
        results['baseline']['errors'].append(error)
        results['baseline']['times'].append(baseline_result['timing']['total'])
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(angles))
    width = 0.35
    
    # 定位误差对比
    axes[0].bar(x - width/2, results['bionic']['errors'], width, label='Proposed Method', color='steelblue')
    axes[0].bar(x + width/2, results['baseline']['errors'], width, label='Baseline Method', color='coral')
    axes[0].set_xlabel('Angle (°)')
    axes[0].set_ylabel('Localization Error (px)')
    axes[0].set_title('Localization Error Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(angles)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 处理时间对比
    axes[1].bar(x - width/2, results['bionic']['times'], width, label='Proposed Method', color='steelblue')
    axes[1].bar(x + width/2, results['baseline']['times'], width, label='Baseline Method', color='coral')
    axes[1].set_xlabel('Angle (°)')
    axes[1].set_ylabel('Processing Time (ms)')
    axes[1].set_title('Processing Time Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(angles)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./method_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印结果
    print("\n" + "="*70)
    print("Method Comparison Results")
    print("="*70)
    print(f"{'Angle':>8} | {'Proposed (px)':>14} | {'Baseline (px)':>14} | {'Improvement':>12}")
    print("-"*70)
    
    for i, angle in enumerate(angles):
        bionic_err = results['bionic']['errors'][i]
        baseline_err = results['baseline']['errors'][i]
        improvement = (baseline_err - bionic_err) / baseline_err * 100 if baseline_err > 0 else 0
        print(f"{angle:>8}° | {bionic_err:>14.2f} | {baseline_err:>14.2f} | {improvement:>11.1f}%")
    
    print("-"*70)
    mean_bionic = np.mean([e for e in results['bionic']['errors'] if e < float('inf')])
    mean_baseline = np.mean([e for e in results['baseline']['errors'] if e < float('inf')])
    print(f"{'Mean':>8} | {mean_bionic:>14.2f} | {mean_baseline:>14.2f} | {(mean_baseline-mean_bionic)/mean_baseline*100:>11.1f}%")
    print("="*70)
    
    return results


def demo_complete_pipeline():
    """演示完整检测流程"""
    from module1_synthetic_data import SyntheticCornerGenerator
    
    generator = SyntheticCornerGenerator(image_size=256)
    detector = BionicCornerDetector()
    
    # 测试不同角度
    angles = [30, 60, 90, 120]
    
    for angle in angles:
        img, corner_pos = generator.generate_single_test_image(angle, add_noise=False)
        result = detector.detect_and_visualize(
            img, 
            ground_truth=corner_pos,
            save_path=f'./detection_result_{angle}deg.png'
        )
        
        print(f"\nAngle: {angle}°")
        print(f"  Corners detected: {result['n_corners']}")
        print(f"  Edge points: {result['statistics']['n_edge_points']}")
        print(f"  Significant points: {result['statistics']['n_significant_points']}")
        print(f"  Total time: {result['timing']['total']:.2f} ms")


if __name__ == "__main__":
    # 演示完整流程
    demo_complete_pipeline()
    
    # 方法对比
    compare_methods()
