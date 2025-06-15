import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from ..config.config import Config, LoggingConfig
from .logger import Logger

class Visualizer:
    """Visualization tool class"""
    
    def __init__(self, config: LoggingConfig):
        """Initialize visualizer"""
        self.config = config
        self.log_dir = config.log_dir
        self.logger = Logger(config)
        self.plot_dir = self.logger.get_log_dir()
        os.makedirs(self.plot_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set basic matplotlib style
        plt.style.use('seaborn')
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
        title: str = "Prediction vs Actual"
    ):
        """绘制预测值与实际值的对比图"""
        plt.figure(figsize=(10, 6))
        plt.scatter(actuals, predictions, alpha=0.5)
        
        # 添加理想预测线
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
        
        # 添加10%误差上下限线
        plt.plot([min_val, max_val], [min_val*0.9, max_val*0.9], 'g--', label='10% Error Lower Bound')
        plt.plot([min_val, max_val], [min_val*1.1, max_val*1.1], 'g--', label='10% Error Upper Bound')
        
        plt.title(title)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)
        
        # 计算准确率（误差在10%以内的比例）
        relative_error = np.abs(predictions - actuals) / np.abs(actuals)
        accuracy = np.mean(relative_error <= 0.1) * 100
        plt.text(0.05, 0.95, f'Accuracy (Error ≤ 10%): {accuracy:.2f}%',
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