from dataclasses import dataclass, field
from typing import List, Optional
import torch
import os

# 获取项目根目录，以下函数的作用等同于"../../../"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    data_path: str = os.path.join(ROOT_DIR, "data", "weather.csv")
    
    # 时间特征
    time_column: str = "Formatted Date"
    
    # 输入特征（根据README说明选择）
    input_features: List[str] = (
        # 数值型特征
        "Temperature (C)",        # 温度
        "Apparent Temperature (C)", # 体感温度
        "Humidity",              # 湿度
        "Wind Speed (km/h)",     # 风速
        "Wind Bearing (degrees)", # 风向
        "Visibility (km)",       # 能见度
        "Pressure (millibars)",  # 气压
        # 分类特征
        "Precip Type"           # 降水类型
    )
    
    # 分类特征（根据README说明）
    categorical_features: List[str] = (
        "Precip Type"  # 降水类型（rain, snow, null）
    )
    
    # 目标特征（预测目标）
    target_features: List[str] = (
        "Temperature (C)",        # 温度
        "Apparent Temperature (C)", # 体感温度
        "Humidity",              # 湿度
        "Wind Speed (km/h)",     # 风速
        "Wind Bearing (degrees)", # 风向
        "Visibility (km)",       # 能见度
        "Pressure (millibars)"   # 气压
    )
    
    # 时间序列参数
    sequence_length: int = 24  # 使用过去24个时间点的数据
    prediction_length: int = 1  # 预测未来1个时间点
    
    # 数据划分比例
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 数据标准化
    normalize: bool = True
    
    # 批处理大小
    batch_size: int = 32

@dataclass
class ModelConfig:
    """模型配置"""
    # 输入维度（特征数量）
    input_dim: int = 10  # 7个数值特征 + 3个分类特征（降水类型的独热编码）
    
    # 模型维度
    d_model: int = 512
    
    # 注意力头数
    nhead: int = 8
    
    # 编码器层数
    num_encoder_layers: int = 6
    
    # 前馈网络维度
    dim_feedforward: int = 2048
    
    # Dropout比率
    dropout: float = 0.1
    
    # 预测长度
    prediction_length: int = 1

@dataclass
class TrainingConfig:
    """训练配置"""
    # 随机种子
    seed: int = 42
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 训练轮数
    epochs: int = 100
    
    # 学习率
    learning_rate: float = 0.001
    
    # 学习率调度器
    use_lr_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # 早停
    early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # 梯度裁剪
    gradient_clip_val: float = 1.0
    
    # 权重衰减
    weight_decay: float = 0.01

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

@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """配置验证"""
        # 确保数据划分比例之和为1
        assert abs(self.data.train_ratio + self.data.val_ratio + self.data.test_ratio - 1.0) < 1e-6, \
            "数据划分比例之和必须为1"
        
        # 确保序列长度合理
        assert self.data.sequence_length > 0, "序列长度必须大于0"
        assert self.data.prediction_length > 0, "预测长度必须大于0"
        
        # 确保批处理大小合理
        assert self.data.batch_size > 0, "批处理大小必须大于0"
        
        # 确保模型参数合理
        assert self.model.d_model > 0, "模型维度必须大于0"
        assert self.model.nhead > 0, "注意力头数必须大于0"
        assert self.model.num_encoder_layers > 0, "编码器层数必须大于0"
        assert 0 <= self.model.dropout <= 1, "Dropout比率必须在0到1之间" 