"""
Module 1: Synthetic Corner Image Dataset Generation
基于简单细胞与端抑制细胞协同处理的仿生弯角检测方法 - 合成角点图像集生成模块

This module generates synthetic corner images with various angles (15°-165°)
for corner detection algorithm evaluation.
"""

import numpy as np
import cv2
import os
from typing import Tuple, List, Dict
import json

class SyntheticCornerGenerator:
    """合成角点图像生成器"""
    
    def __init__(self, image_size: int = 512, output_dir: str = './synthetic_corners'):
        """
        初始化生成器
        
        Parameters:
        -----------
        image_size : int
            图像尺寸 (默认512x512)
        output_dir : str
            输出目录路径
        """
        self.image_size = image_size
        self.output_dir = output_dir
        self.ground_truth = []
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(os.path.join(output_dir, 'uniform'))
            os.makedirs(os.path.join(output_dir, 'noisy'))
    
    def generate_corner_image(self, 
                              angle_deg: float, 
                              corner_pos: Tuple[int, int],
                              edge_length: int = 150,
                              line_thickness: int = 2,
                              background: int = 128,
                              foreground: int = 255) -> np.ndarray:
        """
        生成单个角点图像
        
        Parameters:
        -----------
        angle_deg : float
            角点角度（度）
        corner_pos : tuple
            角点位置 (x, y)
        edge_length : int
            边缘长度
        line_thickness : int
            线条粗细
        background : int
            背景灰度值
        foreground : int
            前景灰度值
            
        Returns:
        --------
        np.ndarray
            生成的角点图像
        """
        img = np.ones((self.image_size, self.image_size), dtype=np.uint8) * background
        
        cx, cy = corner_pos
        half_angle = np.radians(angle_deg / 2)
        
        # 计算两条边的端点
        # 第一条边（向上偏转半角）
        end1_x = int(cx + edge_length * np.cos(np.pi/2 - half_angle))
        end1_y = int(cy - edge_length * np.sin(np.pi/2 - half_angle))
        
        # 第二条边（向上偏转负半角）
        end2_x = int(cx + edge_length * np.cos(np.pi/2 + half_angle))
        end2_y = int(cy - edge_length * np.sin(np.pi/2 + half_angle))
        
        # 绘制两条边
        cv2.line(img, (cx, cy), (end1_x, end1_y), foreground, line_thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (end2_x, end2_y), foreground, line_thickness, cv2.LINE_AA)
        
        return img
    
    def add_gaussian_noise(self, img: np.ndarray, snr_db: float) -> np.ndarray:
        """
        添加高斯噪声
        
        Parameters:
        -----------
        img : np.ndarray
            输入图像
        snr_db : float
            信噪比（dB）
            
        Returns:
        --------
        np.ndarray
            添加噪声后的图像
        """
        img_float = img.astype(np.float64)
        signal_power = np.mean(img_float ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power)
        
        noise = np.random.normal(0, noise_std, img.shape)
        noisy_img = img_float + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        return noisy_img
    
    def generate_dataset(self, 
                         angles: List[float] = None,
                         samples_per_angle: int = 34,
                         snr_range: Tuple[float, float] = (20, 40)) -> Dict:
        """
        生成完整的合成角点数据集
        
        Parameters:
        -----------
        angles : list
            角度列表，默认为15°到165°，步长7.5°
        samples_per_angle : int
            每个角度的样本数
        snr_range : tuple
            信噪比范围 (min_snr, max_snr)
            
        Returns:
        --------
        dict
            数据集统计信息
        """
        if angles is None:
            angles = np.arange(15, 166, 7.5).tolist()
        
        self.ground_truth = []
        total_images = 0
        
        print("Generating Synthetic Corner Dataset...")
        print(f"Angles: {len(angles)} levels from {min(angles)}° to {max(angles)}°")
        print(f"Samples per angle: {samples_per_angle}")
        
        for angle in angles:
            print(f"  Processing angle: {angle}°")
            
            for sample_idx in range(samples_per_angle):
                # 随机位置（确保角点在图像中心区域）
                margin = 100
                cx = np.random.randint(margin + 50, self.image_size - margin - 50)
                cy = np.random.randint(margin + 50, self.image_size - margin - 50)
                
                # 生成均匀背景图像
                img_uniform = self.generate_corner_image(
                    angle_deg=angle,
                    corner_pos=(cx, cy),
                    edge_length=np.random.randint(100, 180),
                    line_thickness=np.random.randint(1, 4)
                )
                
                # 保存均匀背景图像
                filename_uniform = f"corner_angle{int(angle):03d}_sample{sample_idx:03d}_uniform.png"
                cv2.imwrite(os.path.join(self.output_dir, 'uniform', filename_uniform), img_uniform)
                
                # 生成带噪声图像
                snr = np.random.uniform(snr_range[0], snr_range[1])
                img_noisy = self.add_gaussian_noise(img_uniform, snr)
                
                filename_noisy = f"corner_angle{int(angle):03d}_sample{sample_idx:03d}_noisy.png"
                cv2.imwrite(os.path.join(self.output_dir, 'noisy', filename_noisy), img_noisy)
                
                # 记录真值信息
                self.ground_truth.append({
                    'filename_uniform': filename_uniform,
                    'filename_noisy': filename_noisy,
                    'angle': angle,
                    'corner_x': cx,
                    'corner_y': cy,
                    'snr_db': snr,
                    'sample_idx': sample_idx
                })
                
                total_images += 2
        
        # 保存真值文件
        gt_path = os.path.join(self.output_dir, 'ground_truth.json')
        with open(gt_path, 'w') as f:
            json.dump(self.ground_truth, f, indent=2)
        
        stats = {
            'total_images': total_images,
            'uniform_images': total_images // 2,
            'noisy_images': total_images // 2,
            'angle_levels': len(angles),
            'samples_per_angle': samples_per_angle,
            'image_size': self.image_size,
            'output_dir': self.output_dir
        }
        
        print(f"\nDataset generation complete!")
        print(f"Total images: {total_images}")
        print(f"Ground truth saved to: {gt_path}")
        
        return stats
    
    def generate_single_test_image(self, angle: float, add_noise: bool = False, snr: float = 30) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        生成单个测试图像
        
        Parameters:
        -----------
        angle : float
            角度
        add_noise : bool
            是否添加噪声
        snr : float
            信噪比
            
        Returns:
        --------
        tuple
            (图像, 角点真值位置)
        """
        cx, cy = self.image_size // 2, self.image_size // 2
        img = self.generate_corner_image(angle, (cx, cy))
        
        if add_noise:
            img = self.add_gaussian_noise(img, snr)
        
        return img, (cx, cy)


def visualize_sample_corners():
    """可视化示例角点图像"""
    import matplotlib.pyplot as plt
    
    generator = SyntheticCornerGenerator(image_size=256, output_dir='./temp_corners')
    
    angles = [30, 60, 90, 120, 150]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for idx, angle in enumerate(angles):
        img, pos = generator.generate_single_test_image(angle, add_noise=False)
        axes[0, idx].imshow(img, cmap='gray')
        axes[0, idx].plot(pos[0], pos[1], 'r+', markersize=10, markeredgewidth=2)
        axes[0, idx].set_title(f'{angle}° (Uniform)')
        axes[0, idx].axis('off')
        
        img_noisy, _ = generator.generate_single_test_image(angle, add_noise=True, snr=25)
        axes[1, idx].imshow(img_noisy, cmap='gray')
        axes[1, idx].plot(pos[0], pos[1], 'r+', markersize=10, markeredgewidth=2)
        axes[1, idx].set_title(f'{angle}° (SNR=25dB)')
        axes[1, idx].axis('off')
    
    plt.suptitle('Synthetic Corner Images at Different Angles', fontsize=14)
    plt.tight_layout()
    plt.savefig('./synthetic_corner_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 清理临时文件
    import shutil
    if os.path.exists('./temp_corners'):
        shutil.rmtree('./temp_corners')


if __name__ == "__main__":
    # 生成完整数据集
    generator = SyntheticCornerGenerator(
        image_size=512,
        output_dir='./synthetic_corner_dataset'
    )
    
    # 按照论文设置生成数据集
    # 21个角度等级：15°到165°
    angles = list(range(15, 166, 7)) + [165]  # 确保包含165°
    angles = sorted(list(set(angles)))[:21]   # 保留21个角度
    
    stats = generator.generate_dataset(
        angles=angles,
        samples_per_angle=34,
        snr_range=(20, 40)
    )
    
    print("\n" + "="*50)
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 可视化样例
    visualize_sample_corners()
