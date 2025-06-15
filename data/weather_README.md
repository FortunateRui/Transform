# 数据类型
| 字段名 | 类型 | 描述 | 可能的值 | 备注 |
|--------|------|------|------|------|
| Formatted Date | datetime | 格式化的日期时间 | 2006-12-13 02:00:00.000 +0100 | 应作为时间序列的索引，进行位置编码 |
| Summary | string | 天气概况 | Clear | 文本，**不适合**作为输入特征 |
| Precip Type | string | 降水类型 | rain,snow,null | 分类变量，进行独热编码后可作为输入特征 |
| Temperature (C) | float | 温度(摄氏度) | 20.3 | 可作为输入特征 |
| Apparent Temperature (C) | float | 体感温度(摄氏度) | 20.3 | 可作为输入特征 |
| Humidity | float | 湿度 | 0.5 | 可作为输入特征 |
| Wind Speed (km/h) | float | 风速(公里/小时) | 10.2 | 可作为输入特征 |
| Wind Bearing (degrees) | float | 风向(度) | 180 | 可作为输入特征 |
| Visibility (km) | float | 能见度(公里) | 10.2 | 可作为输入特征 |
| Loud Cover | float | 噪音覆盖 | 0.5 | 可作为输入特征 |
| Pressure (millibars) | float | 气压(毫巴) | 1013.25 | 可作为输入特征 |
| Daily Summary | string | 每日天气总结 | Partly cloudy starting in the afternoon. | 文本，**不适合**作为输入特征 |
