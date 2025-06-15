import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
        self.categorical_encoders = {}  # 存储分类变量的编码器
        
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
        
        # 处理分类变量
        processed_features = []
        for feature in features:
            if feature in self.config.data.categorical_features:
                # 对分类变量进行独热编码
                if feature not in self.categorical_encoders:
                    self.categorical_encoders[feature] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = self.categorical_encoders[feature].fit_transform(df[[feature]])
                else:
                    encoded = self.categorical_encoders[feature].transform(df[[feature]])
                processed_features.append(encoded)
            else:
                # 数值型特征直接使用
                processed_features.append(df[[feature]].values)
        
        # 合并所有特征
        data = np.hstack(processed_features)
        
        # 标准化数值型特征
        if self.config.data.normalize:
            data = self.scaler.fit_transform(data)
            
        return data, {
            "scaler": self.scaler,
            "categorical_encoders": self.categorical_encoders
        }
    
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
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=4,  # 每个GPU使用4个工作进程
            pin_memory=True,  # 使用固定内存加速数据传输
            persistent_workers=True,  # 保持工作进程存活
            prefetch_factor=2  # 预加载因子
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        return train_loader, val_loader, test_loader
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """反标准化数据"""
        if self.config.data.normalize:
            return self.scaler.inverse_transform(data)
        return data
    
    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备序列数据
        
        参数:
            df (pd.DataFrame): 预处理后的数据框
            
        返回:
            Tuple[np.ndarray, np.ndarray]: 输入序列和目标序列
        """
        # 获取特征列
        feature_columns = [col for col in df.columns if col not in ['time', 'timezone']]
        
        # 准备序列数据
        X, y = [], []
        for i in range(len(df) - self.sequence_length - self.prediction_length + 1):
            # 输入序列
            X.append(df[feature_columns].iloc[i:i+self.sequence_length].values)
            # 目标序列 - 只使用温度
            y.append(df['Temperature (C)'].iloc[i+self.sequence_length:i+self.sequence_length+self.prediction_length].values)
            
        return np.array(X), np.array(y)
    
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