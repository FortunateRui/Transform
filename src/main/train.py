import os
import argparse
import torch
import numpy as np
import random
import torch.optim as optim
from ..config.config import Config
from ..models.time_series_transformer import TimeSeriesTransformer
from ..utils.data_processor import DataProcessor
from ..utils.logger import Logger
from ..utils.visualizer import Visualizer
from ..utils.trainer import Trainer
from datetime import datetime

def set_seed(seed: int):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from (e.g., 20250615_210905/latest_model)')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    try:
        # 加载配置
        config = Config()
        logger = Logger(config)
        
        # 设置随机种子
        set_seed(config.training.seed)
        
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
        model.to(config.training.device)
        
        # 创建优化器
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # 创建学习率调度器
        scheduler = None
        if config.training.use_lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.training.scheduler_factor,
                patience=config.training.scheduler_patience
            )
        
        # 创建数据加载器
        data_processor = DataProcessor(config)
        train_loader, val_loader, test_loader = data_processor.prepare_data()
        
        # 创建可视化器
        visualizer = Visualizer(config, logger)
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            logger=logger,
            visualizer=visualizer
        )
        
        # 设置优化器和调度器
        trainer.optimizer = optimizer
        trainer.scheduler = scheduler
        
        # 设置模型保存目录
        if args.resume:
            # 从指定检查点恢复
            checkpoint_path = os.path.join(config.logging.model_dir, args.resume + ".pth")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # 加载检查点
            checkpoint = torch.load(checkpoint_path)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if trainer.scheduler and 'scheduler_state_dict' in checkpoint:
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            trainer.train_losses = checkpoint['train_losses']
            trainer.val_losses = checkpoint['val_losses']
            trainer.best_val_loss = checkpoint['best_val_loss']
            logger.log_info(f"Resumed from checkpoint: {checkpoint_path}")
            
            # 创建新的时间戳目录
            trainer.model_save_dir = os.path.join(config.logging.model_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(trainer.model_save_dir, exist_ok=True)
        else:
            # 创建新的时间戳目录
            trainer.model_save_dir = os.path.join(config.logging.model_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(trainer.model_save_dir, exist_ok=True)
            start_epoch = 0
        
        # 开始训练
        trainer.train(start_epoch=start_epoch)
        
    except Exception as e:
        logger.log_error(f"训练过程中出现错误: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 