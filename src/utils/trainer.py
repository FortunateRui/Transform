import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import signal
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
from ..config.config import Config
from .logger import Logger
from .visualizer import Visualizer
import os
from datetime import datetime

class Trainer:
    """通用训练器"""
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        logger: Logger,
        visualizer: Visualizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader
    ):
        self.model = model
        self.config = config
        self.logger = logger
        self.visualizer = visualizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 设置设备
        self.device = torch.device(config.training.device)
        self.model.to(self.device)
        self.logger.log_info(f"使用设备: {self.device}")
        if self.device.type == 'cuda':
            self.logger.log_info(f"GPU型号: {torch.cuda.get_device_name(0)}")
            self.logger.log_info(f"可用GPU数量: {torch.cuda.device_count()}")
        
        # 初始化优化器
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # 初始化学习率调度器
        if config.training.use_lr_scheduler:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.training.scheduler_factor,
                patience=config.training.scheduler_patience
            )
        
        # 初始化损失函数 - 温度预测使用MSE
        self.criterion = nn.MSELoss()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
        # 设置中断处理
        self.setup_interrupt_handler()
        
    def setup_interrupt_handler(self):
        """设置中断处理"""
        def signal_handler(signum, frame):
            self.logger.log_warning("训练被中断，正在保存模型和日志...")
            self.save_checkpoint(is_interrupted=True)
            self.visualize_training()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            if self.config.training.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_val
                )
                
            self.optimizer.step()
            total_loss += loss.item()
            
            # 计算准确率（误差在10%以内算正确）
            relative_error = torch.abs(output - target) / torch.abs(target)
            correct = (relative_error <= 0.1).float().mean()
            total_correct += correct.item() * target.size(0)
            total_samples += target.size(0)
            
            # 记录训练指标
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                self.logger.log_metrics(
                    {
                        'train_loss': loss.item(),
                        'temperature_mae': torch.mean(torch.abs(output - target)).item(),
                        'train_accuracy': correct.item()
                    },
                    self.current_epoch * len(self.train_loader) + batch_idx
                )
                
        return total_loss / len(self.train_loader), total_correct / total_samples
    
    def validate(self) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                total_mae += torch.mean(torch.abs(output - target)).item()
                
                # 计算准确率（误差在20%以内算正确）
                relative_error = torch.abs(output - target) / torch.abs(target)
                correct = (relative_error <= 0.2).float().mean()
                total_correct += correct.item() * target.size(0)
                total_samples += target.size(0)
                
        avg_loss = total_loss / len(self.val_loader)
        avg_mae = total_mae / len(self.val_loader)
        accuracy = total_correct / total_samples
        
        # 记录验证指标
        self.logger.log_metrics(
            {
                'val_loss': avg_loss,
                'temperature_mae': avg_mae,
                'val_accuracy': accuracy
            },
            self.current_epoch
        )
                
        return avg_loss, accuracy
    
    def save_checkpoint(self, is_interrupted: bool = False):
        """保存检查点"""
        # 如果不是中断状态，且不是最后一个epoch，则根据保存频率决定是否保存
        if not is_interrupted and self.current_epoch < self.config.training.epochs - 1:
            if self.current_epoch % self.config.logging.save_frequency != 0:
                return
                
        try:
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss
            }
            
            if self.config.training.use_lr_scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            # 保存最新模型
            if self.config.logging.save_latest_model or is_interrupted:
                latest_path = os.path.join(self.model_save_dir, "latest_model.pth")
                torch.save(checkpoint, latest_path)
                self.logger.log_info(f"已保存最新模型到: {latest_path}")
            
            # 保存最佳模型
            if self.config.logging.save_best_model and not is_interrupted:
                if self.val_losses[-1] < self.best_val_loss:
                    self.best_val_loss = self.val_losses[-1]
                    best_path = os.path.join(self.model_save_dir, "best_model.pth")
                    torch.save(checkpoint, best_path)
                    self.logger.log_info(f"已保存最佳模型到: {best_path}")
                    
        except Exception as e:
            self.logger.log_error(f"保存检查点时出错: {str(e)}")
            # 如果保存失败，尝试保存到备份文件
            try:
                backup_dir = os.path.join(self.model_save_dir, "backup")
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, f"checkpoint_epoch_{self.current_epoch}.pth")
                torch.save(checkpoint, backup_path)
                self.logger.log_info(f"已保存备份检查点到: {backup_path}")
            except Exception as backup_e:
                self.logger.log_error(f"保存备份检查点也失败: {str(backup_e)}")
                
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.config.training.use_lr_scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
    def visualize_training(self):
        """可视化训练过程"""
        self.visualizer.plot_loss_curves(
            self.train_losses,
            self.val_losses
        )
        
    def train(self, start_epoch: int = 0):
        """训练模型"""
        self.current_epoch = start_epoch
        self.logger.log_info("开始训练...")
        
        try:
            for epoch in range(start_epoch, self.config.training.epochs):
                self.current_epoch = epoch
                
                # 训练一个epoch
                train_loss, train_accuracy = self.train_epoch()
                self.train_losses.append(train_loss)
                
                # 验证
                if epoch % self.config.logging.val_every_n_epochs == 0:
                    val_loss, val_accuracy = self.validate()
                    self.val_losses.append(val_loss)
                    
                    # 更新学习率
                    if self.config.training.use_lr_scheduler:
                        self.scheduler.step(val_loss)
                        
                    # 记录epoch结果
                    self.logger.log_epoch(
                        epoch,
                        {'loss': train_loss, 'accuracy': train_accuracy},
                        {'loss': val_loss, 'accuracy': val_accuracy}
                    )
                    
                    # 保存检查点
                    self.save_checkpoint()
                    
                # 可视化训练过程
                if epoch % self.config.logging.plot_every_n_epochs == 0:
                    self.visualize_training()
                    
        except KeyboardInterrupt:
            self.logger.log_warning("训练被中断")
        finally:
            self.save_checkpoint(is_interrupted=True)
            self.visualize_training()
            self.logger.log_info("训练结束") 