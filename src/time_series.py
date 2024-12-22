import sys
sys.path.append('../')  # 如果util模块在上级目录
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
# import pmdarima as pm

trace_path = "../../device_11.csv"



data = pd.read_csv(trace_path, names=["device_id", "opcode", "offset", "length", "timestamp"])
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='us')
data['hour'] = data['timestamp'].dt.floor('h')


data['io_size'] = data['length']
hourly_io = data.groupby('hour')['io_size'].mean()


hourly_io = hourly_io.asfreq('h')
train_size = int(len(hourly_io) * 0.8)
train, test = hourly_io[:train_size], hourly_io[train_size:]

# model = ARIMA(hourly_io, order=(12, 1, 8))  
# model = ARIMA(hourly_io, order=(14, 1, 8))  
# model = ARIMA(hourly_io, order=(14, 1, 15))  
model = ARIMA(hourly_io, order=(20, 1, 15))  

# model = ARIMA(hourly_io, order=(15, 1, 8))  
# model = ARIMA(hourly_io, order=(16, 1, 8))  

model_fit = model.fit()



predictions = model_fit.forecast(steps=len(test))

# 结果可视化
plt.plot(train.index, train, label='Actual Data')
# plt.plot(hourly_io.index, hourly_io, label='Actual Data')
plt.plot(test.index, predictions, label='Predicted Data', color='red')
plt.title('IO Size Evolution with Time')
plt.xlabel('Time')
plt.ylabel('IO size/byte')
plt.legend()
plt.show()