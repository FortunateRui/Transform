import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import seaborn as sns
from datetime import datetime
from ..config.config import Config
from .logger import Logger

class Visualizer:
    """可视化工具类"""
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.plot_dir = logger.get_log_dir()  # 使用logger的日志目录
        os.makedirs(self.plot_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 设置绘图风格
        sns.set_style("whitegrid")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 设置中文字体
        plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
    def plot_loss_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_accuracies: Optional[List[float]] = None,
        val_accuracies: Optional[List[float]] = None
    ):
        """
        绘制损失曲线和准确率曲线
        
        参数:
            train_losses (List[float]): 训练损失列表
            val_losses (List[float]): 验证损失列表
            train_accuracies (Optional[List[float]]): 训练准确率列表
            val_accuracies (Optional[List[float]]): 验证准确率列表
        """
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 绘制损失曲线
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='训练损失')
        ax1.plot(epochs, val_losses, 'r-', label='验证损失')
        ax1.set_title('损失曲线')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        # 如果有准确率数据，绘制准确率曲线
        if train_accuracies is not None and val_accuracies is not None:
            ax2.plot(epochs, train_accuracies, 'b-', label='训练准确率')
            ax2.plot(epochs, val_accuracies, 'r-', label='验证准确率')
            ax2.set_title('准确率曲线 (误差≤10%)')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('准确率')
            ax2.set_ylim(0, 1)  # 准确率范围从0到1
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_curves.png'))
        plt.close()
        
    def plot_metrics(
        self,
        metrics: Dict[str, List[float]],
        title: str = "训练指标"
    ):
        """绘制多个指标曲线"""
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(next(iter(metrics.values()))) + 1)
        
        for metric_name, values in metrics.items():
            plt.plot(epochs, values, label=metric_name)
            
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        save_path = os.path.join(self.plot_dir, f'metrics_{self.timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_prediction_vs_actual(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        title: str = "预测值 vs 实际值"
    ):
        """
        绘制预测值与实际值的对比图
        
        参数:
            predictions (np.ndarray): 预测值数组
            actuals (np.ndarray): 实际值数组
            title (str): 图表标题
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        
        # 添加误差范围线
        plt.plot([actuals.min(), actuals.max()], 
                [actuals.min() * 0.9, actuals.max() * 0.9], 'g--', lw=1, 
                label='10%误差下限')
        plt.plot([actuals.min(), actuals.max()], 
                [actuals.min() * 1.1, actuals.max() * 1.1], 'g--', lw=1, 
                label='10%误差上限')
        
        plt.title(title)
        plt.xlabel('实际温度 (°C)')
        plt.ylabel('预测温度 (°C)')
        plt.legend()
        plt.grid(True)
        
        # 计算并显示准确率
        relative_error = np.abs(predictions - actuals) / np.abs(actuals)
        accuracy = np.mean(relative_error <= 0.1) * 100
        plt.text(0.05, 0.95, f'准确率: {accuracy:.2f}%', 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'prediction_vs_actual.png'))
        plt.close() 