"""
Main Experiment: Bionic Corner Detection Based on Cooperative Processing 
                 of Simple Cells and End-Stopped Cells
基于简单细胞与端抑制细胞协同处理的仿生弯角检测方法 - 主实验脚本

This script runs the complete experiment as described in the paper,
including dataset generation, algorithm evaluation, and result visualization.

Usage:
    python main_experiment.py

Author: Research Team
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime

# 导入各模块
from module1_synthetic_data import SyntheticCornerGenerator
from module2_simple_cell import SimpleCellModel
from module3_end_stopped_cell import EndStoppedCellModel
from module4_corner_localization import CornerLocalizer
from module5_corner_detector import BionicCornerDetector, BaselineCornerDetector
from module6_evaluation import PerformanceEvaluator, PsychophysicalValidator


class ExperimentRunner:
    """
    实验运行器
    
    执行论文中描述的完整实验流程
    """
    
    def __init__(self, output_dir: str = './experiment_results'):
        """
        初始化实验运行器
        
        Parameters:
        -----------
        output_dir : str
            输出目录
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
        
        # 实验配置
        self.config = {
            'image_size': 512,
            'angles': [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165],
            'samples_per_angle': 34,
            'n_eval_samples': 20,
            'snr_range': (20, 40),
            
            # 简单细胞参数（表1）
            'gabor_wavelength': 6.0,
            'gabor_n_orientations': 12,
            'edge_threshold_percentile': 75,
            
            # 端抑制细胞参数
            'log_sigma': 4.0,
            'response_threshold_percentile': 80,
            
            # 角点定位参数（表2）
            'cluster_distance_threshold': 15.0,
            'min_cluster_size': 3,
            'max_pairing_distance': 40.0
        }
        
        # 结果存储
        self.results = {}
    
    def save_config(self):
        """保存实验配置"""
        config_path = os.path.join(self.output_dir, 'data', 'experiment_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to: {config_path}")
    
    def run_experiment_1_dataset_generation(self):
        """
        实验1：生成合成角点数据集
        
        对应论文3.1节
        """
        print("\n" + "="*70)
        print("EXPERIMENT 1: Synthetic Corner Dataset Generation")
        print("="*70)
        
        generator = SyntheticCornerGenerator(
            image_size=self.config['image_size'],
            output_dir=os.path.join(self.output_dir, 'data', 'synthetic_corners')
        )
        
        stats = generator.generate_dataset(
            angles=self.config['angles'],
            samples_per_angle=self.config['samples_per_angle'],
            snr_range=self.config['snr_range']
        )
        
        self.results['dataset'] = stats
        
        # 生成示例图像
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        sample_angles = [15, 30, 60, 90, 120, 150]
        
        for idx, angle in enumerate(sample_angles):
            img_uniform, pos = generator.generate_single_test_image(angle, add_noise=False)
            img_noisy, _ = generator.generate_single_test_image(angle, add_noise=True, snr=25)
            
            axes[0, idx].imshow(img_uniform, cmap='gray')
            axes[0, idx].plot(pos[0], pos[1], 'r+', markersize=10, markeredgewidth=2)
            axes[0, idx].set_title(f'{angle}° (Uniform)')
            axes[0, idx].axis('off')
            
            axes[1, idx].imshow(img_noisy, cmap='gray')
            axes[1, idx].plot(pos[0], pos[1], 'r+', markersize=10, markeredgewidth=2)
            axes[1, idx].set_title(f'{angle}° (Noisy)')
            axes[1, idx].axis('off')
        
        plt.suptitle('Synthetic Corner Dataset Samples', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figures', 'fig3_dataset_samples.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return stats
    
    def run_experiment_2_algorithm_components(self):
        """
        实验2：算法各组件可视化
        
        对应论文第2章方法部分的图示
        """
        print("\n" + "="*70)
        print("EXPERIMENT 2: Algorithm Components Visualization")
        print("="*70)
        
        generator = SyntheticCornerGenerator(image_size=256)
        
        # 创建各组件
        simple_cell = SimpleCellModel(
            wavelength=self.config['gabor_wavelength'],
            n_orientations=self.config['gabor_n_orientations']
        )
        
        end_stopped_cell = EndStoppedCellModel(
            sigma_s=self.config['log_sigma'],
            response_threshold_percentile=self.config['response_threshold_percentile']
        )
        
        corner_localizer = CornerLocalizer(
            distance_threshold=self.config['cluster_distance_threshold'],
            min_cluster_size=self.config['min_cluster_size'],
            max_pairing_distance=self.config['max_pairing_distance']
        )
        
        # 图3：Gabor滤波器组
        simple_cell.visualize_gabor_bank(
            save_path=os.path.join(self.output_dir, 'figures', 'fig3_gabor_filters.png')
        )
        
        # 测试角度
        test_angle = 60
        img, corner_pos = generator.generate_single_test_image(test_angle)
        
        # 处理流程
        edge_mag, edge_ori = simple_cell.compute_edge_response(img)
        edge_points = simple_cell.extract_edge_points(edge_mag, self.config['edge_threshold_percentile'])
        sig_points, responses = end_stopped_cell.dynamic_construct(img, edge_points, edge_ori)
        corners = corner_localizer.localize_corners(sig_points, responses)
        
        # 图2：算法流程可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 输入图像
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].plot(corner_pos[0], corner_pos[1], 'r+', markersize=15, markeredgewidth=3)
        axes[0, 0].set_title('(a) Input Image')
        axes[0, 0].axis('off')
        
        # 边缘响应
        axes[0, 1].imshow(edge_mag, cmap='hot')
        axes[0, 1].set_title('(b) Edge Magnitude E(x,y)')
        axes[0, 1].axis('off')
        
        # 边缘方向
        axes[0, 2].imshow(edge_ori, cmap='hsv')
        axes[0, 2].set_title('(c) Edge Orientation Θ(x,y)')
        axes[0, 2].axis('off')
        
        # 边缘点
        edge_img = np.zeros_like(img)
        for pt in edge_points:
            edge_img[pt[0], pt[1]] = 255
        axes[1, 0].imshow(edge_img, cmap='gray')
        axes[1, 0].set_title(f'(d) Edge Points (N={len(edge_points)})')
        axes[1, 0].axis('off')
        
        # 端抑制细胞响应
        axes[1, 1].imshow(img, cmap='gray', alpha=0.5)
        if len(sig_points) > 0:
            sc = axes[1, 1].scatter(sig_points[:, 1], sig_points[:, 0],
                                   c=responses, cmap='hot', s=30)
            plt.colorbar(sc, ax=axes[1, 1], fraction=0.046)
        axes[1, 1].set_title(f'(e) End-Stopped Response (N={len(sig_points)})')
        axes[1, 1].axis('off')
        
        # 角点定位结果
        axes[1, 2].imshow(img, cmap='gray')
        if len(corners) > 0:
            for corner in corners:
                pos = corner['position']
                axes[1, 2].plot(pos[1], pos[0], 's', color='lime', markersize=15,
                               markeredgecolor='black', markeredgewidth=2)
        axes[1, 2].plot(corner_pos[0], corner_pos[1], 'r+', markersize=15, markeredgewidth=3)
        axes[1, 2].set_title('(f) Corner Localization Result')
        axes[1, 2].axis('off')
        
        plt.suptitle('Algorithm Pipeline Visualization', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figures', 'fig2_algorithm_pipeline.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Algorithm components visualization completed.")
    
    def run_experiment_3_performance_evaluation(self):
        """
        实验3：性能评估
        
        对应论文3.2和3.3节
        """
        print("\n" + "="*70)
        print("EXPERIMENT 3: Performance Evaluation")
        print("="*70)
        
        # 创建检测器
        bionic_detector = BionicCornerDetector(
            gabor_wavelength=self.config['gabor_wavelength'],
            gabor_n_orientations=self.config['gabor_n_orientations'],
            edge_threshold_percentile=self.config['edge_threshold_percentile'],
            log_sigma=self.config['log_sigma'],
            response_threshold_percentile=self.config['response_threshold_percentile'],
            cluster_distance_threshold=self.config['cluster_distance_threshold'],
            min_cluster_size=self.config['min_cluster_size'],
            max_pairing_distance=self.config['max_pairing_distance']
        )
        
        baseline_detector = BaselineCornerDetector(log_sigma=self.config['log_sigma'])
        
        # 评估本方法
        print("\n[1/2] Evaluating Proposed Method...")
        evaluator_bionic = PerformanceEvaluator()
        results_bionic = evaluator_bionic.evaluate_dataset(
            bionic_detector,
            angles=self.config['angles'],
            samples_per_angle=self.config['n_eval_samples']
        )
        
        # 评估基线方法
        print("[2/2] Evaluating Baseline Method...")
        evaluator_baseline = PerformanceEvaluator()
        results_baseline = evaluator_baseline.evaluate_dataset(
            baseline_detector,
            angles=self.config['angles'],
            samples_per_angle=self.config['n_eval_samples']
        )
        
        # 生成表4：不同角度条件下的平均定位误差对比
        table4_data = []
        for angle in self.config['angles']:
            bionic_data = results_bionic[results_bionic['angle'] == angle]['localization_error']
            baseline_data = results_baseline[results_baseline['angle'] == angle]['localization_error']
            
            bionic_mle = bionic_data[bionic_data < float('inf')].mean()
            baseline_mle = baseline_data[baseline_data < float('inf')].mean()
            bionic_rmse = np.sqrt((bionic_data[bionic_data < float('inf')]**2).mean())
            baseline_rmse = np.sqrt((baseline_data[baseline_data < float('inf')]**2).mean())
            improvement = (baseline_mle - bionic_mle) / baseline_mle * 100 if baseline_mle > 0 else 0
            
            table4_data.append({
                'Angle': angle,
                'Proposed_MLE': round(bionic_mle, 2),
                'Baseline_MLE': round(baseline_mle, 2),
                'Proposed_RMSE': round(bionic_rmse, 2),
                'Baseline_RMSE': round(baseline_rmse, 2),
                'Improvement': f"{improvement:.1f}%"
            })
        
        table4 = pd.DataFrame(table4_data)
        table4.to_csv(os.path.join(self.output_dir, 'data', 'table4_angle_comparison.csv'), index=False)
        
        # 图6：热力图（角点响应强度空间分布）
        self._generate_heatmap_figure(bionic_detector)
        
        # 图7：箱线图
        self._generate_boxplot_figure(results_bionic, results_baseline)
        
        # 保存结果
        self.results['performance'] = {
            'bionic': evaluator_bionic.compute_metrics(),
            'baseline': evaluator_baseline.compute_metrics()
        }
        
        # 生成报告
        report = evaluator_bionic.generate_report(
            os.path.join(self.output_dir, 'reports', 'performance_report.txt')
        )
        print("\n" + report)
        
        return table4
    
    def _generate_heatmap_figure(self, detector):
        """生成图6：热力图"""
        generator = SyntheticCornerGenerator(image_size=512)
        angles = [30, 60, 90, 120, 150]
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        for idx, angle in enumerate(angles):
            # 收集多个样本的响应分布
            response_map = np.zeros((101, 101))
            n_samples = 10
            
            for _ in range(n_samples):
                img, corner_pos = generator.generate_single_test_image(angle)
                result = detector.detect(img, return_intermediate=True)
                
                sig_points = result['intermediate']['significant_points']
                
                for pt in sig_points:
                    # 计算相对于角点的偏移
                    dy = int(pt[0] - corner_pos[1]) + 50
                    dx = int(pt[1] - corner_pos[0]) + 50
                    
                    if 0 <= dy < 101 and 0 <= dx < 101:
                        response_map[dy, dx] += 1
            
            # 归一化
            response_map = response_map / response_map.max() if response_map.max() > 0 else response_map
            
            im = axes[idx].imshow(response_map, cmap='hot', extent=[-50, 50, 50, -50])
            axes[idx].plot(0, 0, 'g+', markersize=15, markeredgewidth=2)
            axes[idx].set_title(f'{angle}°')
            axes[idx].set_xlabel('X offset (px)')
            if idx == 0:
                axes[idx].set_ylabel('Y offset (px)')
            plt.colorbar(im, ax=axes[idx], fraction=0.046)
        
        plt.suptitle('Figure 6: Corner Response Intensity Spatial Distribution Heatmaps', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figures', 'fig6_heatmaps.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_boxplot_figure(self, results_bionic, results_baseline):
        """生成图7：箱线图"""
        # 按角度分组
        acute = (15, 75)
        right = (75, 105)
        obtuse = (105, 165)
        
        def filter_by_range(df, low, high):
            mask = (df['angle'] >= low) & (df['angle'] <= high)
            return df[mask & (df['localization_error'] < float('inf'))]['localization_error'].values
        
        bionic_acute = filter_by_range(results_bionic, *acute)
        bionic_right = filter_by_range(results_bionic, *right)
        bionic_obtuse = filter_by_range(results_bionic, *obtuse)
        
        baseline_acute = filter_by_range(results_baseline, *acute)
        baseline_right = filter_by_range(results_baseline, *right)
        baseline_obtuse = filter_by_range(results_baseline, *obtuse)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 本方法
        bp1 = axes[0].boxplot([bionic_acute, bionic_right, bionic_obtuse],
                             labels=['Acute\n(15°-75°)', 'Right\n(75°-105°)', 'Obtuse\n(105°-165°)'],
                             patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
        axes[0].set_ylabel('Localization Error (px)')
        axes[0].set_title('Proposed Method')
        axes[0].grid(True, alpha=0.3)
        
        # 基线方法
        bp2 = axes[1].boxplot([baseline_acute, baseline_right, baseline_obtuse],
                             labels=['Acute\n(15°-75°)', 'Right\n(75°-105°)', 'Obtuse\n(105°-165°)'],
                             patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('coral')
            patch.set_alpha(0.7)
        axes[1].set_ylabel('Localization Error (px)')
        axes[1].set_title('Baseline Method')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Figure 7: Localization Error Distribution by Angle Group', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figures', 'fig7_boxplots.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_experiment_4_psychophysical_validation(self):
        """
        实验4：心理物理学合理性验证
        
        对应论文3.4节
        """
        print("\n" + "="*70)
        print("EXPERIMENT 4: Psychophysical Validation")
        print("="*70)
        
        # 创建检测器
        bionic_detector = BionicCornerDetector(
            gabor_wavelength=self.config['gabor_wavelength'],
            gabor_n_orientations=self.config['gabor_n_orientations'],
            edge_threshold_percentile=self.config['edge_threshold_percentile'],
            log_sigma=self.config['log_sigma'],
            response_threshold_percentile=self.config['response_threshold_percentile'],
            cluster_distance_threshold=self.config['cluster_distance_threshold'],
            min_cluster_size=self.config['min_cluster_size'],
            max_pairing_distance=self.config['max_pairing_distance']
        )
        
        baseline_detector = BaselineCornerDetector(log_sigma=self.config['log_sigma'])
        
        # 心理物理学验证
        validator = PsychophysicalValidator()
        
        print("\n[1/2] Computing detection deviations...")
        validator.compute_detection_deviation(bionic_detector, n_samples=30)
        
        print("[2/2] Computing correlations...")
        correlation = validator.compute_correlation()
        
        # 生成表7：检测偏移量与人类感知偏移量对照
        table7 = validator.generate_comparison_table()
        table7.to_csv(os.path.join(self.output_dir, 'data', 'table7_deviation_comparison.csv'), index=False)
        
        # 图9：散点图
        # 图10：趋势图
        validator.visualize_correlation(
            save_path=os.path.join(self.output_dir, 'figures', 'fig9_10_correlation.png')
        )
        
        # 生成表8：生理学合理性评估指标对比
        table8_data = [{
            'Method': 'Proposed Method',
            'Pearson_r': round(correlation['pearson_r'], 3),
            'SSIM': round(correlation['ssim'], 3),
            'NMSE': round(correlation['nmse'], 3),
            'Composite_Score': round(correlation['composite_score'], 3)
        }]
        table8 = pd.DataFrame(table8_data)
        table8.to_csv(os.path.join(self.output_dir, 'data', 'table8_physiological_metrics.csv'), index=False)
        
        # 保存结果
        self.results['psychophysical'] = correlation
        
        # 生成报告
        report = validator.generate_report()
        with open(os.path.join(self.output_dir, 'reports', 'psychophysical_report.txt'), 'w') as f:
            f.write(report)
        
        print("\n" + report)
        
        return correlation
    
    def run_all_experiments(self):
        """运行所有实验"""
        start_time = time.time()
        
        print("\n" + "#"*70)
        print("# BIONIC CORNER DETECTION EXPERIMENT")
        print("# Based on Cooperative Processing of Simple Cells and End-Stopped Cells")
        print("#"*70)
        print(f"\nExperiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_dir}")
        
        # 保存配置
        self.save_config()
        
        # 运行各实验
        self.run_experiment_1_dataset_generation()
        self.run_experiment_2_algorithm_components()
        table4 = self.run_experiment_3_performance_evaluation()
        correlation = self.run_experiment_4_psychophysical_validation()
        
        # 生成最终报告
        self._generate_final_report()
        
        total_time = time.time() - start_time
        
        print("\n" + "#"*70)
        print("# EXPERIMENT COMPLETED")
        print("#"*70)
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Results saved to: {self.output_dir}")
        print(f"\nKey Results:")
        print(f"  - MLE (Proposed): {self.results['performance']['bionic']['MLE']:.3f} px")
        print(f"  - MLE (Baseline): {self.results['performance']['baseline']['MLE']:.3f} px")
        print(f"  - Pearson Correlation: {self.results['psychophysical']['pearson_r']:.4f}")
        print(f"  - Composite Score: {self.results['psychophysical']['composite_score']:.4f}")
    
    def _generate_final_report(self):
        """生成最终实验报告"""
        report = []
        report.append("="*70)
        report.append("BIONIC CORNER DETECTION - FINAL EXPERIMENT REPORT")
        report.append("="*70)
        report.append(f"\nExperiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Output Directory: {self.output_dir}")
        report.append("")
        
        report.append("\n1. EXPERIMENT CONFIGURATION")
        report.append("-"*40)
        for key, value in self.config.items():
            report.append(f"   {key}: {value}")
        
        report.append("\n2. PERFORMANCE METRICS")
        report.append("-"*40)
        if 'performance' in self.results:
            bionic = self.results['performance']['bionic']
            baseline = self.results['performance']['baseline']
            
            report.append(f"   Proposed Method:")
            report.append(f"     - MLE: {bionic['MLE']:.3f} px")
            report.append(f"     - RMSE: {bionic['RMSE']:.3f} px")
            report.append(f"     - Detection Rate: {bionic['detection_rate']:.1f}%")
            report.append(f"     - Mean Time: {bionic['mean_time']:.2f} ms")
            
            report.append(f"   Baseline Method:")
            report.append(f"     - MLE: {baseline['MLE']:.3f} px")
            report.append(f"     - RMSE: {baseline['RMSE']:.3f} px")
            report.append(f"     - Detection Rate: {baseline['detection_rate']:.1f}%")
            
            improvement = (baseline['MLE'] - bionic['MLE']) / baseline['MLE'] * 100
            report.append(f"   Improvement: {improvement:.1f}%")
        
        report.append("\n3. PSYCHOPHYSICAL VALIDATION")
        report.append("-"*40)
        if 'psychophysical' in self.results:
            psy = self.results['psychophysical']
            report.append(f"   Pearson Correlation: r = {psy['pearson_r']:.4f}")
            report.append(f"   SSIM: {psy['ssim']:.4f}")
            report.append(f"   NMSE: {psy['nmse']:.4f}")
            report.append(f"   Composite Score: {psy['composite_score']:.4f}")
        
        report.append("\n4. OUTPUT FILES")
        report.append("-"*40)
        report.append("   Figures:")
        report.append("     - fig2_algorithm_pipeline.png")
        report.append("     - fig3_gabor_filters.png")
        report.append("     - fig3_dataset_samples.png")
        report.append("     - fig6_heatmaps.png")
        report.append("     - fig7_boxplots.png")
        report.append("     - fig9_10_correlation.png")
        report.append("   Data:")
        report.append("     - table4_angle_comparison.csv")
        report.append("     - table7_deviation_comparison.csv")
        report.append("     - table8_physiological_metrics.csv")
        report.append("     - experiment_config.json")
        
        report.append("\n" + "="*70)
        
        report_text = "\n".join(report)
        
        with open(os.path.join(self.output_dir, 'reports', 'final_report.txt'), 'w') as f:
            f.write(report_text)
        
        print(report_text)


def main():
    """主函数"""
    # 创建实验运行器
    runner = ExperimentRunner(output_dir='./experiment_results')
    
    # 运行所有实验
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
