import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from ..config.config import Config
from ..models.time_series_transformer import TimeSeriesTransformer
from ..utils.data_processor import DataProcessor
from ..utils.logger import Logger

def load_model(config: Config, model_path: str) -> TimeSeriesTransformer:
    """加载模型"""
    model = TimeSeriesTransformer(
        input_dim=config.model.input_dim,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_encoder_layers=config.model.num_encoder_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        prediction_length=config.model.prediction_length
    )
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=config.training.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.training.device)
    model.eval()
    return model

def prepare_prediction_data(df: pd.DataFrame, data_processor: DataProcessor) -> np.ndarray:
    """准备预测数据"""
    # 将时间列转换为datetime类型，并统一转换为UTC时间
    df[data_processor.config.data.time_column] = pd.to_datetime(
        df[data_processor.config.data.time_column],
        utc=True
    )
    
    # 按时间排序
    df = df.sort_values(data_processor.config.data.time_column)
    
    # 预处理数据
    data, _ = data_processor.preprocess_data(df)
    return data

def predict_temperature(
    model: TimeSeriesTransformer,
    data: np.ndarray,
    sequence_length: int,
    device: str,
    logger: Logger
) -> np.ndarray:
    """预测温度"""
    predictions = []
    total_samples = len(data) - sequence_length
    
    logger.log_info(f"开始预测，总样本数: {total_samples}")
    
    with torch.no_grad():
        for i in tqdm(range(sequence_length, len(data)), desc="预测进度"):
            # 获取输入序列
            x = data[i-sequence_length:i]
            x = torch.FloatTensor(x).unsqueeze(0).to(device)  # 添加batch维度
            
            # 预测
            output = model(x)
            predictions.append(output.cpu().numpy()[0, 0])  # 只取温度预测值
            
            # # 每1000个样本记录一次进度
            # if (i - sequence_length) % 1000 == 0:
            #     logger.log_info(f"已处理 {i - sequence_length}/{total_samples} 个样本")
    
    return np.array(predictions)

def main():
    # 创建配置
    config = Config()
    
    # 创建日志记录器
    logger = Logger(config)
    logger.log_info("开始预测过程...")
    
    try:
        # 创建输出文件
        output_path = os.path.join(
            os.path.dirname(config.data.data_path),
            f'weather_prediction_with_temperature_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        logger.log_info(f"创建输出文件: {output_path}")
        
        # 加载数据
        logger.log_info("加载预测数据...")
        df = pd.read_csv(config.data.data_path.replace('weather.csv', 'weather_prediction.csv'))
        logger.log_info(f"加载完成，数据形状: {df.shape}")
        
        # 创建数据处理器
        data_processor = DataProcessor(config)
        
        # 准备预测数据
        logger.log_info("准备预测数据...")
        data = prepare_prediction_data(df, data_processor)
        logger.log_info(f"数据预处理完成，形状: {data.shape}")
        
        # 加载模型
        logger.log_info("加载模型...")
        model_path = os.path.join(config.logging.model_dir, "predict", "model.pth")
        model = load_model(config, model_path)
        logger.log_info("模型加载完成")
        
        # 进行预测
        logger.log_info("开始预测...")
        predictions = predict_temperature(
            model,
            data,
            config.data.sequence_length,
            config.training.device,
            logger
        )
        logger.log_info(f"预测完成，预测结果形状: {predictions.shape}")
        
        # 将预测结果添加到DataFrame中
        logger.log_info("将预测结果添加到数据中...")
        df['prediction_temperature'] = np.nan
        
        # 转换回实际温度值
        # 创建一个与训练时相同维度的数组，其他特征填充为0
        full_predictions = np.zeros((len(predictions), len(config.data.input_features)))
        full_predictions[:, 0] = predictions  # 将温度预测值放在第一列
        
        actual_predictions = data_processor.inverse_transform(full_predictions)[:, 0]  # 只取温度列
        
        # 填充实际温度值
        df.loc[config.data.sequence_length:, 'prediction_temperature'] = actual_predictions
        
        # 保存结果
        logger.log_info("保存预测结果...")
        df.to_csv(output_path, index=False)
        logger.log_info(f"预测完成，结果已保存至: {output_path}")
        
        # 输出一些统计信息
        valid_predictions = df['prediction_temperature'].dropna()
        logger.log_info(f"预测结果统计:")
        logger.log_info(f"总样本数: {len(df)}")
        logger.log_info(f"有效预测数: {len(valid_predictions)}")
        
        logger.log_info(f"预测温度范围: {valid_predictions.min():.2f}°C 到 {valid_predictions.max():.2f}°C")
        logger.log_info(f"预测温度均值: {valid_predictions.mean():.2f}°C")
        logger.log_info(f"预测温度标准差: {valid_predictions.std():.2f}°C")
        
        # 计算预测误差
        actual_temperatures = df['Temperature (C)'].iloc[config.data.sequence_length:].values
        mae = np.mean(np.abs(actual_predictions - actual_temperatures))
        mse = np.mean((actual_predictions - actual_temperatures) ** 2)
        rmse = np.sqrt(mse)
        
        logger.log_info(f"预测误差统计:")
        logger.log_info(f"平均绝对误差 (MAE): {mae:.2f}°C")
        logger.log_info(f"均方误差 (MSE): {mse:.2f}°C")
        logger.log_info(f"均方根误差 (RMSE): {rmse:.2f}°C")
        
    except Exception as e:
        logger.log_error(f"预测过程中出现错误: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 