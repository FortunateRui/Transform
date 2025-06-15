import os
import argparse
import torch
import numpy as np
import random
from ..config.config import Config
from ..models.time_series_transformer import TimeSeriesTransformer
from ..utils.data_processor import DataProcessor
from ..utils.logger import Logger
from ..utils.visualizer import Visualizer
from ..utils.trainer import Trainer

def set_seed(seed: int):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练时间序列预测模型')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='检查点文件路径，用于继续训练'
    )
    return parser.parse_args()

def main():
    """主函数"""
    # 加载配置
    config = Config()
    
    # 设置随机种子
    set_seed(config.training.seed)
    
    # 初始化日志记录器
    logger = Logger(config)
    
    try:
        # 加载数据
        data_processor = DataProcessor(config)
        train_loader, val_loader, test_loader = data_processor.prepare_data()
        
        # 创建模型
        model = TimeSeriesTransformer(
            input_dim=config.model.input_dim,
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            num_encoder_layers=config.model.num_encoder_layers,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            prediction_length=config.model.prediction_length
        )
        
        # 初始化可视化器
        visualizer = Visualizer(config, logger)
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            config=config,
            logger=logger,
            visualizer=visualizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        # 开始训练
        trainer.train()
        
    except Exception as e:
        logger.log_error(f"训练过程中出现错误: {str(e)}")
        raise e

if __name__ == '__main__':
    main() 