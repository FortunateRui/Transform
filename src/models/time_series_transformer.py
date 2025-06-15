import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
import math

class TimeSeriesTransformer(nn.Module):
    """
    用于时间序列预测的Transformer模型
    
    属性:
        input_dim (int): 输入特征维度
        d_model (int): Transformer模型维度
        nhead (int): 注意力头数
        num_encoder_layers (int): 编码器层数
        dim_feedforward (int): 前馈网络维度
        dropout (float): Dropout比率
        prediction_length (int): 预测序列长度
    """
    
    def __init__(
        self,
        # 输入特征维度，输入序列每个时间点的特征数量，例如多种技术指标
        input_dim: int,
        # Transformer模型维度，决定模型能够捕捉的特征数量，需要可以被注意力头数整除
        d_model: int = 512,
        # 注意力头数，决定模型能够并行处理的注意力头数量
        nhead: int = 8,
        # 编码器层数，层数越多模型表达能力越强，训练难度也越高
        num_encoder_layers: int = 6,
        # 前馈网络维度，一般设置为模型维度的4倍
        dim_feedforward: int = 2048,
        # Dropout比率，用于防止过拟合，值如0.1表示10%的节点随机失活
        dropout: float = 0.1,
        # 预测序列长度，决定模型能够预测的未来时间步数
        prediction_length: int = 1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.prediction_length = prediction_length
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # 输出投影层 - 只输出温度预测值
        self.output_projection = nn.Linear(d_model, prediction_length)
        
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            src (torch.Tensor): 输入序列，形状为 [batch_size, seq_len, input_dim]
            src_mask (Optional[torch.Tensor]): 源序列的掩码
            
        返回:
            torch.Tensor: 温度预测结果，形状为 [batch_size, prediction_length]
        """
        # 输入投影
        x = self.input_projection(src)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x, src_mask)
        
        # 只使用最后一个时间步的输出进行预测
        x = x[:, -1, :]
        
        # 输出投影 - 只预测温度
        output = self.output_projection(x)
        
        return output

class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]
            
        返回:
            torch.Tensor: 添加位置编码后的张量
        """
        # 确保位置编码的长度足够
        if x.size(1) > self.pe.size(0):
            # 如果序列长度超过最大长度，动态生成位置编码
            position = torch.arange(x.size(1)).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
            pe = torch.zeros(x.size(1), 1, self.d_model, device=x.device)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            x = x + pe.transpose(0, 1)
        else:
            x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

def create_mask(seq_len: int) -> torch.Tensor:
    """
    创建Transformer的掩码矩阵
    
    参数:
        seq_len (int): 序列长度
        
    返回:
        torch.Tensor: 掩码矩阵
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask 