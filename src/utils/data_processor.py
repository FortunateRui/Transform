import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from ..config.config import Config

class WeatherDataset(Dataset):
    """天气数据集类"""
    def __init__(self, data: np.ndarray, sequence_length: int, prediction_length: int):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.prediction_length + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取输入序列（所有特征）
        x = self.data[idx:idx + self.sequence_length]
        # 获取目标序列（只取温度，即第一个特征）
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.prediction_length, 0]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class DataProcessor:
    """数据处理器"""
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        
    def load_data(self) -> pd.DataFrame:
        """加载数据"""
        df = pd.read_csv(self.config.data.data_path)
        
        # 清理列名（去除前后空格）
        df.columns = df.columns.str.strip()
        
        # 验证所需的列是否存在
        missing_columns = [col for col in self.config.data.input_features if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据文件中缺少以下列: {missing_columns}")
            
        # 将时间列转换为datetime类型，并统一转换为UTC时间
        df[self.config.data.time_column] = pd.to_datetime(
            df[self.config.data.time_column],
            utc=True
        )
        
        # 按时间排序
        df = df.sort_values(self.config.data.time_column)
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """数据预处理"""
        # 选择特征
        features = self.config.data.input_features
        
        # 提取数值特征
        data = df[list(features)].values  # 将特征元组转换为列表
        
        # 标准化数值型特征
        if self.config.data.normalize:
            data = self.scaler.fit_transform(data)
            print(f"特征标准化完成，特征均值: {self.scaler.mean_}, 标准差: {self.scaler.scale_}")
            
        return data, {"scaler": self.scaler}
    
    def create_datasets(self, data: np.ndarray) -> Tuple[Dataset, Dataset, Dataset]:
        """创建训练、验证和测试数据集"""
        # 计算划分点
        n = len(data)
        train_size = int(n * self.config.data.train_ratio)
        val_size = int(n * self.config.data.val_ratio)
        
        # 划分数据
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        print(f"数据集划分完成:")
        print(f"训练集: {len(train_data)} 样本")
        print(f"验证集: {len(val_data)} 样本")
        print(f"测试集: {len(test_data)} 样本")
        
        # 创建数据集
        train_dataset = WeatherDataset(
            train_data,
            self.config.data.sequence_length,
            self.config.data.prediction_length
        )
        val_dataset = WeatherDataset(
            val_data,
            self.config.data.sequence_length,
            self.config.data.prediction_length
        )
        test_dataset = WeatherDataset(
            test_data,
            self.config.data.sequence_length,
            self.config.data.prediction_length
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """反标准化数据"""
        if self.config.data.normalize:
            return self.scaler.inverse_transform(data)
        return data
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        准备训练、验证和测试数据
        
        返回:
            Tuple[DataLoader, DataLoader, DataLoader]: 训练、验证和测试数据加载器
        """
        # 加载数据
        df = self.load_data()
        
        # 预处理数据
        data, scaler_info = self.preprocess_data(df)
        
        # 创建数据集
        train_dataset, val_dataset, test_dataset = self.create_datasets(data)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_dataloaders(
            train_dataset, val_dataset, test_dataset
        )
        
        return train_loader, val_loader, test_loader 