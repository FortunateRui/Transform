import os
import logging
from datetime import datetime
from typing import Dict, Any
import json
from ..config.config import Config

class Logger:
    """日志记录器"""
    def __init__(self, config: Config):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建日志目录
        self.log_dir = os.path.join(config.logging.log_dir, self.timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建模型保存目录
        self.model_dir = os.path.join(config.logging.model_dir, self.timestamp)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 配置日志记录器
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器，指定UTF-8编码
        log_file = os.path.join(self.log_dir, 'training.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 保存配置
        self.save_config()
        
    def save_config(self):
        """保存配置到日志目录"""
        config_dict = {
            'data': self.config.data.__dict__,
            'model': self.config.model.__dict__,
            'training': self.config.training.__dict__,
            'logging': self.config.logging.__dict__
        }
        
        with open(os.path.join(self.log_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
            
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """记录指标"""
        metrics_str = ' - '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.logger.info(f'Step {step}: {metrics_str}')
        
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """记录每个epoch的结果"""
        train_str = ' - '.join([f'train_{k}: {v:.4f}' for k, v in train_metrics.items()])
        val_str = ' - '.join([f'val_{k}: {v:.4f}' for k, v in val_metrics.items()])
        self.logger.info(f'Epoch {epoch}: {train_str} | {val_str}')
        
    def log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(message)
        
    def log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(message)
        
    def log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(message)
        
    def get_timestamp(self) -> str:
        """获取时间戳"""
        return self.timestamp
        
    def get_log_dir(self) -> str:
        """获取日志目录"""
        return self.log_dir 