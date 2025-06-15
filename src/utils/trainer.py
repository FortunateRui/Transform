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
        
        # 启用多GPU训练
        if torch.cuda.device_count() > 1:
            self.logger.log_info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
            self.model = nn.DataParallel(self.model)
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
            try:
                self.save_checkpoint()  # 保存最新模型
                self.visualize_training()  # 保存训练可视化
            except Exception as e:
                self.logger.log_error(f"保存中断状态时出错: {str(e)}")
            sys.exit(0)
            
        # 注册信号处理器
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_correct = 0
        total_samples = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # 计算损失
            loss = self.criterion(output, target)
            
            # 计算MAE
            mae = torch.mean(torch.abs(output - target))
            
            # 计算准确率（误差在20%以内算正确）
            relative_error = torch.abs(output - target) / (torch.abs(target) + 1e-6)  # 添加小量避免除零
            correct = (relative_error <= 0.2).float().mean()
            
            # 缩放损失以支持梯度累积
            loss = loss / self.config.training.gradient_accumulation_steps
            loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.config.training.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 累积统计信息
            total_loss += loss.item() * self.config.training.gradient_accumulation_steps
            total_mae += mae.item()
            total_correct += correct.item() * target.size(0)
            total_samples += target.size(0)
            batch_count += 1
            
            # 打印训练进度
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                metrics = {
                    'loss': loss.item() * self.config.training.gradient_accumulation_steps,
                    'mae': mae.item(),
                    'accuracy': correct.item(),
                    'prediction': output[0].item(),  # 第一个样本的预测值
                    'target': target[0].item()      # 第一个样本的真实值
                }
                self.logger.log_metrics(metrics, batch_idx)
        
        # 计算平均指标
        avg_loss = total_loss / batch_count
        avg_mae = total_mae / batch_count
        avg_accuracy = total_correct / total_samples
        
        return avg_loss, avg_mae, avg_accuracy
    
    def validate(self):
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
                
                # 计算损失
                loss = self.criterion(output, target)
                
                # 计算MAE
                mae = torch.mean(torch.abs(output - target))
                
                # 计算准确率（误差在20%以内算正确）
                relative_error = torch.abs(output - target) / (torch.abs(target) + 1e-6)
                correct = (relative_error <= 0.2).float().mean()
                
                total_loss += loss.item()
                total_mae += mae.item()
                total_correct += correct.item() * target.size(0)
                total_samples += target.size(0)
        
        # 计算平均指标
        avg_loss = total_loss / len(self.val_loader)
        avg_mae = total_mae / len(self.val_loader)
        avg_accuracy = total_correct / total_samples
        
        return avg_loss, avg_mae, avg_accuracy
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        # 保存最新模型
        latest_path = os.path.join(self.config.logging.log_dir, 'latest_model.pth')
        torch.save(checkpoint, latest_path)
        self.logger.log_info(f"已保存最新模型到: {latest_path}")
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = os.path.join(self.config.logging.log_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.log_info(f"已保存最佳模型到: {best_path}")
            
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        self.logger.log_info(f"已加载检查点: {checkpoint_path}")
        
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
                train_loss, train_mae, train_accuracy = self.train_epoch()
                self.train_losses.append(train_loss)
                
                # 验证
                if epoch % self.config.logging.val_every_n_epochs == 0:
                    val_loss, val_mae, val_accuracy = self.validate()
                    self.val_losses.append(val_loss)
                    
                    # 更新学习率调度器
                    if self.scheduler is not None:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(val_loss)  # 使用验证损失来调整学习率
                        else:
                            self.scheduler.step()
                    
                    # 保存最佳模型
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)
                        self.logger.log_info(f"保存最佳模型，验证损失: {val_loss:.4f}")
                    
                    # 记录训练和验证指标
                    self.logger.log_epoch(
                        epoch,
                        {'loss': train_loss, 'mae': train_mae, 'accuracy': train_accuracy},
                        {'loss': val_loss, 'mae': val_mae, 'accuracy': val_accuracy}
                    )
                
                # 保存最新模型
                if self.config.logging.save_latest_model:
                    self.save_checkpoint()
                
                # 早停检查
                if self.config.training.early_stopping:
                    if self.early_stopping_counter >= self.config.training.early_stopping_patience:
                        self.logger.log_info("触发早停机制，停止训练")
                        break
                        
        except Exception as e:
            self.logger.log_error(f"训练过程中出现错误: {str(e)}")
            # 保存当前状态
            self.save_checkpoint()  # 保存最新模型
            self.visualize_training()  # 保存训练可视化
            raise e 