import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional
from ..config.config import Config, LoggingConfig
from .logger import Logger

class Visualizer:
    """Visualization tool class"""
    
    def __init__(self, config: Config, logger: Logger):
        """Initialize visualizer"""
        self.config = config
        self.logger = logger
        self.log_dir = logger.log_dir
        
        # Create plot directory
        self.plot_dir = os.path.join(self.log_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set basic matplotlib style
        sns.set_style("whitegrid")
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
    def plot_loss_curves(
        self,
        train_losses: list,
        val_losses: list,
        title: str = "Training and Validation Loss"
    ):
        """绘制损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        save_path = os.path.join(self.plot_dir, f"loss_curves_{self.timestamp}.png")
        plt.savefig(save_path)
        plt.close()
        
    def plot_metrics(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Training Metrics"
    ):
        """绘制多个指标曲线"""
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(next(iter(metrics.values()))) + 1)
        
        for metric_name, values in metrics.items():
            plt.plot(epochs, values, label=metric_name)
            
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
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
        title: str = "Prediction vs Actual"
    ):
        """绘制预测值与实际值的对比图"""
        plt.figure(figsize=(10, 6))
        plt.scatter(actuals, predictions, alpha=0.5)
        
        # 添加理想预测线
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
        
        # 添加20%误差上下限线
        plt.plot([min_val, max_val], [min_val*0.8, max_val*0.8], 'g--', label='20% Error Lower Bound')
        plt.plot([min_val, max_val], [min_val*1.2, max_val*1.2], 'g--', label='20% Error Upper Bound')
        
        plt.title(title)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)
        
        # 计算准确率（误差在20%以内的比例）
        relative_error = np.abs(predictions - actuals) / np.abs(actuals)
        accuracy = np.mean(relative_error <= 0.2) * 100
        plt.text(0.05, 0.95, f'Accuracy (Error ≤ 20%): {accuracy:.2f}%',
                transform=plt.gca().transAxes, verticalalignment='top')
        
        # 保存图表
        save_path = os.path.join(self.plot_dir, f"prediction_vs_actual_{self.timestamp}.png")
        plt.savefig(save_path)
        plt.close()
        
    def plot_temperature_prediction(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        title: str = "Temperature Prediction"
    ):
        """绘制温度预测结果"""
        plt.figure(figsize=(12, 6))
        plt.plot(actuals, label='Actual Temperature', color='blue')
        plt.plot(predictions, label='Predicted Temperature', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Time Step')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        save_path = os.path.join(self.plot_dir, f"temperature_prediction_{self.timestamp}.png")
        plt.savefig(save_path)
        plt.close()

    def plot_training_curves(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """Plot training curves"""
        plt.figure(figsize=(12, 5))
        
        # Plot loss curves
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_curves.png'))
        plt.close()
    
    def plot_error_distribution(self, errors):
        """Plot error distribution"""
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, density=True)
        plt.title('Prediction Error Distribution')
        plt.xlabel('Relative Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Add 8% error lines
        plt.axvline(x=0.08, color='r', linestyle='--', label='8% Error Line')
        plt.axvline(x=-0.08, color='r', linestyle='--')
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(self.plot_dir, 'error_distribution.png'))
        plt.close() 