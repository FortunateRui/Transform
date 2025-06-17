import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
import math
import torch.nn.functional as F

class ProbMask(nn.Module):
    def __init__(self, B, H, L_Q, index, scores, device="cpu"):
        super(ProbMask, self).__init__()
        _mask = torch.ones(L_Q, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L_Q, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                           torch.arange(H)[None, :, None],
                           index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
        
    @property
    def mask(self):
        return self._mask

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        # 添加线性变换层
        self.W_q = nn.Linear(512, 512)  # 假设输入维度为512
        self.W_k = nn.Linear(512, 512)
        self.W_v = nn.Linear(512, 512)
        self.W_o = nn.Linear(512, 512)
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1])
        else:
            assert(L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B)[:, None, None],
                  torch.arange(H)[None, :, None],
                  index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys=None, values=None, attn_mask=None):
        """
        前向传播
        Args:
            queries: 查询张量 [B, L, D]
            keys: 键张量 [B, L, D]
            values: 值张量 [B, L, D]
            attn_mask: 注意力掩码
        """
        B, L, D = queries.shape
        if keys is None:
            keys = queries
        if values is None:
            values = queries
            
        # 线性变换
        queries = self.W_q(queries)  # [B, L, D]
        keys = self.W_k(keys)        # [B, L, D]
        values = self.W_v(values)    # [B, L, D]
        
        # 重塑维度
        queries = queries.view(B, L, 1, D).transpose(1, 2)  # [B, 1, L, D]
        keys = keys.view(B, L, 1, D).transpose(1, 2)        # [B, 1, L, D]
        values = values.view(B, L, 1, D).transpose(1, 2)    # [B, 1, L, D]
        
        # 计算稀疏注意力
        U_part = self.factor * np.ceil(np.log(L)).astype('int').item()
        u = self.factor * np.ceil(np.log(L)).astype('int').item()
        
        U_part = U_part if U_part < L else L
        u = u if u < L else L
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        
        # 添加缩放因子
        scale = self.scale or 1./np.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
            
        # 获取上下文
        context = self._get_initial_context(values, L)
        
        # 使用选定的top_k查询更新上下文
        context, attn = self._update_context(context, values, scores_top, index, L, attn_mask)
        
        # 输出投影
        context = context.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        output = self.W_o(context)  # [B, L, D]
        
        return output, attn

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, attn_mask=None):
        """
        前向传播
        Args:
            x: 输入张量 [B, L, D]
            attn_mask: 注意力掩码
        """
        # 自注意力层
        attn_output, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = self.dropout(attn_output)
        x = self.norm1(x + attn_output)
        
        # 前馈网络
        # 转置以进行卷积操作
        x_t = x.transpose(1, 2)  # [B, D, L]
        ff_output = self.conv1(x_t)  # [B, d_ff, L]
        ff_output = self.activation(ff_output)
        ff_output = self.dropout(ff_output)
        ff_output = self.conv2(ff_output)  # [B, D, L]
        # 转置回原始维度
        ff_output = ff_output.transpose(1, 2)  # [B, L, D]
        
        x = self.norm2(x + ff_output)
        
        return x, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

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
    
    def __init__(self, config):
        super(TimeSeriesTransformer, self).__init__()
        
        # 输入投影层
        self.input_projection = nn.Linear(config.model.input_dim, config.model.d_model)
        
        # 创建注意力层
        attention = ProbAttention(
            mask_flag=True,
            factor=5,
            scale=None,
            attention_dropout=0.1,
            output_attention=False
        )
        
        # 创建编码器层
        self.encoder = Encoder(
            [
                EncoderLayer(
                    attention=attention,
                    d_model=config.model.d_model,
                    d_ff=config.model.d_ff,
                    dropout=config.model.dropout,
                    activation=config.model.activation
                ) for _ in range(config.model.n_layers)
            ],
            norm_layer=nn.LayerNorm(config.model.d_model)
        )
        
        # 创建预测层
        self.projection = nn.Linear(config.model.d_model, config.model.output_dim, bias=True)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [B, L, D]
        """
        # 输入投影
        x = self.input_projection(x)  # [B, L, d_model]
        
        # 编码器
        x, _ = self.encoder(x)
        
        # 预测
        output = self.projection(x)
        
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