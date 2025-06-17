import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import sys
import signal
from typing import Tuple, List
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import Config
from src.models.time_series_transformer import TimeSeriesTransformer
from src.utils.data_processor import DataProcessor
from src.utils.logger import Logger
from src.utils.visualizer import Visualizer

# 全局变量用于存储训练状态
training_state = {
    'model': None,
    'optimizer': None,
    'epoch': 0,
    'train_loss': 0.0,
    'val_loss': 0.0,
    'logger': None
}

def signal_handler(signum, frame):
    """处理Ctrl+C中断信号"""
    if training_state['model'] is not None and training_state['logger'] is not None:
        logger = training_state['logger']
        print("\n")  # 确保提示信息在新行显示
        logger.log_info("检测到训练中断，正在保存最新模型...")
        
        try:
            # 保存最新模型
            latest_model_path = os.path.join(logger.model_dir, "latest_model.pth")
            torch.save({
                'epoch': training_state['epoch'],
                'model_state_dict': training_state['model'].state_dict(),
                'optimizer_state_dict': training_state['optimizer'].state_dict(),
                'train_loss': training_state['train_loss'],
                'val_loss': training_state['val_loss'],
            }, latest_model_path)
            
            logger.log_info(f"模型已保存到: {latest_model_path}")
            logger.log_info(f"当前训练状态:")
            logger.log_info(f"- Epoch: {training_state['epoch']}")
            logger.log_info(f"- 训练损失: {training_state['train_loss']:.4f}")
            logger.log_info(f"- 验证损失: {training_state['val_loss']:.4f}")
        except Exception as e:
            logger.log_error(f"保存模型时出错: {str(e)}")
    
    logger.log_info("训练已中断")
    sys.exit(0)

def save_latest_model(model, optimizer, epoch, train_loss, val_loss, logger):
    """保存最新模型"""
    latest_model_path = os.path.join(logger.model_dir, "latest_model.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, latest_model_path)

def train_epoch(
    model: TimeSeriesTransformer,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: str,
    logger: Logger
) -> Tuple[float, float]:
    """
    训练一个epoch
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        logger: 日志器
    Returns:
        train_loss: 训练损失
        train_accuracy: 训练准确率
    """
    model.train()
    total_loss = 0
    total_samples = 0
    correct_predictions = 0
    
    # 创建进度条
    pbar = tqdm(train_loader, desc="训练进度", leave=True)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        output = model(data)  # [B, L, 1]
        
        # 只使用最后一个时间步的预测
        output = output[:, -1, :]  # [B, 1]
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率（相对误差小于8%视为正确）
        with torch.no_grad():
            relative_error = torch.abs(output - target) / torch.abs(target)
            correct_predictions += (relative_error <= 0.08).sum().item()
        
        # 更新统计信息
        total_loss += loss.item() * len(data)
        total_samples += len(data)
        
        # 更新进度条信息
        current_loss = total_loss / total_samples
        current_accuracy = correct_predictions / total_samples
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_accuracy:.4f}'
        })
    
    # 计算平均损失和准确率
    train_loss = total_loss / total_samples
    train_accuracy = correct_predictions / total_samples
    
    return train_loss, train_accuracy

def validate(
    model: TimeSeriesTransformer,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    logger: Logger
) -> Tuple[float, float, List[float]]:
    """验证模型"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    all_errors = []
    
    # 创建进度条
    pbar = tqdm(val_loader, desc="验证进度", leave=True)
    
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output[:, -1, :]  # 只使用最后一个时间步的预测
            
            # 计算损失
            loss = criterion(output, target)
            total_loss += loss.item() * len(data)
            
            # 计算准确率（相对误差小于8%视为正确）
            relative_error = torch.abs(output - target) / torch.abs(target)
            correct_predictions += (relative_error <= 0.08).sum().item()
            total_samples += len(data)
            
            # 收集误差
            all_errors.extend(relative_error.cpu().numpy().flatten())
            
            # 更新进度条信息
            current_loss = total_loss / total_samples
            current_accuracy = correct_predictions / total_samples
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_accuracy:.4f}'
            })
    
    # 计算平均损失和准确率
    val_loss = total_loss / total_samples
    val_accuracy = correct_predictions / total_samples
    
    return val_loss, val_accuracy, all_errors

def main():
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建配置
    config = Config()
    
    # 创建日志记录器
    logger = Logger(config)
    logger.log_info("开始训练过程...")
    
    try:
        # 创建数据处理器
        data_processor = DataProcessor(config)
        
        # 加载和预处理数据
        logger.log_info("加载和预处理数据...")
        df = data_processor.load_data()
        data, preprocess_info = data_processor.preprocess_data(df)
        
        # 准备序列数据
        logger.log_info("准备序列数据...")
        train_dataset, val_dataset, test_dataset = data_processor.create_datasets(data)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False
        )
        
        # 创建模型
        logger.log_info("创建模型...")
        model = TimeSeriesTransformer(config)
        model = model.to(config.training.device)
        
        # 创建损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
            eta_min=config.training.learning_rate * 0.1
        )
        
        # 创建可视化器
        visualizer = Visualizer(config, logger)
        
        # 训练循环
        logger.log_info("开始训练...")
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        # 初始化训练状态
        training_state.update({
            'model': model,
            'optimizer': optimizer,
            'epoch': 0,
            'train_loss': 0.0,
            'val_loss': 0.0,
            'logger': logger
        })
        
        for epoch in range(config.training.epochs):
            logger.log_info(f"\nEpoch {epoch + 1}/{config.training.epochs}")
            
            # 训练
            train_loss, train_accuracy = train_epoch(
                model, train_loader, criterion, optimizer,
                config.training.device, logger
            )
            
            # 验证
            val_loss, val_accuracy, val_errors = validate(
                model, val_loader, criterion,
                config.training.device, logger
            )
            
            # 更新学习率
            scheduler.step()
            
            # 记录指标
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            # 更新训练状态
            training_state.update({
                'model': model,
                'optimizer': optimizer,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'logger': logger
            })
            
            # 记录日志
            logger.log_info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )
            
            # 保存最新模型
            save_latest_model(model, optimizer, epoch, train_loss, val_loss, logger)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(logger.model_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, best_model_path)
                logger.log_info("保存最佳模型")
            
            # 可视化训练过程
            visualizer.plot_training_curves(
                train_losses, val_losses,
                train_accuracies, val_accuracies
            )
            
            # 绘制误差分布
            visualizer.plot_error_distribution(val_errors)
        
        logger.log_info("训练完成！")
        
    except Exception as e:
        logger.log_error(f"训练过程中出现错误: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 