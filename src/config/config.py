from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import os

# 获取项目根目录，以下函数的作用等同于"../../../"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class DataConfig:
    """数据配置"""
    # 数据文件路径
    data_path: str = "data/weather.csv"
    # 时间列名
    time_column: str = "Formatted Date"
    # 输入特征
    input_features: Tuple[str, ...] = (
        "Temperature (C)",
        "Apparent Temperature (C)",
        "Humidity",
        "Wind Speed (km/h)"
    )
    # 目标特征
    target_features: Tuple[str, ...] = ("Temperature (C)",)
    # 序列长度
    sequence_length: int = 24
    # 预测长度
    prediction_length: int = 1
    # 是否标准化
    normalize: bool = True
    # 训练集比例
    train_ratio: float = 0.7
    # 验证集比例
    val_ratio: float = 0.15
    # 测试集比例
    test_ratio: float = 0.15

@dataclass
class ModelConfig:
    """模型配置"""
    # 输入特征维度
    input_dim: int = 4
    # 模型维度
    d_model: int = 512
    # 前馈网络维度
    d_ff: int = 2048
    # 编码器层数
    n_layers: int = 3
    # Dropout比率
    dropout: float = 0.1
    # 激活函数
    activation: str = "relu"
    # 输出维度
    output_dim: int = 1

@dataclass
class TrainingConfig:
    """训练配置"""
    # 训练设备
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    # 训练轮数
    epochs: int = 100
    # 学习率
    learning_rate: float = 0.001
    # 权重衰减
    weight_decay: float = 0.01
    # 批次大小
    batch_size: int = 32
    # 是否使用学习率调度器
    use_lr_scheduler: bool = True
    # 学习率调度器参数
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    # 随机种子
    seed: int = 42

@dataclass
class LoggingConfig:
    """日志配置"""
    # 基础目录
    log_dir: str = os.path.join(ROOT_DIR, "logs")
    model_dir: str = os.path.join(ROOT_DIR, "pth")
    plot_dir: str = os.path.join(ROOT_DIR, "plots")  # 图表保存目录
    
    # 实验名称
    experiment_name: str = "weather_prediction"
    
    # 日志频率（每N个batch记录一次）
    log_every_n_steps: int = 100
    
    # 验证频率（每N个epoch验证一次）
    val_every_n_epochs: int = 1
    
    # 每多少个epoch绘制一次图表
    plot_every_n_epochs: int = 1
    
    # 模型保存配置
    save_latest_model: bool = True  # 是否保存最新模型
    save_best_model: bool = True    # 是否保存最佳模型
    best_model_metric: str = "val_loss"  # 用于判断最佳模型的指标
    
    # 新增参数
    save_frequency: int = 5  # 每多少个epoch保存一次检查点

@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """初始化后的检查"""
        # 检查数据路径
        assert os.path.exists(self.data.data_path), f"数据文件不存在: {self.data.data_path}"
        
        # 检查数据划分比例
        assert abs(self.data.train_ratio + self.data.val_ratio + self.data.test_ratio - 1.0) < 1e-6, \
            "训练集、验证集和测试集的比例之和必须为1"
        
        # 检查模型参数
        assert self.model.input_dim == len(self.data.input_features), \
            f"模型输入维度 ({self.model.input_dim}) 与输入特征数量 ({len(self.data.input_features)}) 不匹配"
        assert self.model.output_dim == len(self.data.target_features), \
            f"模型输出维度 ({self.model.output_dim}) 与目标特征数量 ({len(self.data.target_features)}) 不匹配"
        
        # 检查训练参数
        assert self.training.batch_size > 0, "batch_size必须大于0"
        assert self.training.learning_rate > 0, "learning_rate必须大于0"
        assert self.training.epochs > 0, "epochs必须大于0"
        
        # 创建基础目录
        os.makedirs(self.logging.log_dir, exist_ok=True)
        os.makedirs(self.logging.model_dir, exist_ok=True) 