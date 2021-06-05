# !/bin/python
import akshare as ak
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, GRU
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lib.GRU_module import GRU_module
import os
import matplotlib.pyplot as plt
import math

def get_data_set(stock_id):
    today_date = datetime.date.today()
    start_date = today_date - datetime.timedelta(days=3000) # 往前推3000天
    stock_zh_a_daily_qfq_df = ak.stock_zh_a_daily(symbol=stock_id, start_date=start_date, end_date=today_date, adjust="qfq")  # 历史行情(前复权)
    return stock_zh_a_daily_qfq_df

def training_module(gru, train_set, x_test, y_test):
    x_train, y_train = [],[]
    for i in range(60, len(train_set)):
        x_train.append(train_set[i - 60:i, 0])
        y_train.append(train_set[i, 0])
    gru.train_module(x_train,y_train, x_test, y_test, 60, 1)

def test_predicted_price(gru, test_set, x_test):
    # 测试集输入模型进行预测
    predicted_stock_price = gru.predict(x_test, 60, 1)
    # 对真实数据还原---从（0，1）反归一化到原始范围
    real_stock_price = gru.inverse_norm_data(test_set[60:])
    print(real_stock_price - predicted_stock_price)
    # 画出真实数据和预测数据的对比曲线
    plt.plot(real_stock_price, color='red', label='MaoTai Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted MaoTai Stock Price')
    plt.title('MaoTai Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('MaoTai Stock Price')
    plt.legend()
    plt.show()
    ##########evaluate##############
    # calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
    mse = mean_squared_error(predicted_stock_price, real_stock_price)
    # calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
    rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
    # calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
    mae = mean_absolute_error(predicted_stock_price, real_stock_price)
    print('均方误差: %.6f' % mse)
    print('均方根误差: %.6f' % rmse)
    print('平均绝对误差: %.6f' % mae)

if __name__=='__main__':
    stock_zh_a_daily_qfq_df = get_data_set("sz002044")
    data_len = len(stock_zh_a_daily_qfq_df)
    train_set = stock_zh_a_daily_qfq_df.iloc[0:data_len - 300, 4:5].values  # 留300天当测试集合
    test_set = stock_zh_a_daily_qfq_df.iloc[data_len - 300:, 4:5].values  # 前300天当测试

    gru = GRU_module()

    train_set_scaled = gru.fit_norm_data(train_set)
    test_set = gru.norm_data(test_set) # 用训练好的归一化处理
    x_test, y_test = [], []
    for i in range(60, len(test_set)):
        x_test.append(test_set[i - 60:i, 0])
        y_test.append(test_set[i, 0])
    training_module(gru, train_set_scaled, x_test, y_test)
    test_predicted_price(gru, test_set, x_test)
    # 正式预测
    test_set = stock_zh_a_daily_qfq_df.iloc[data_len - 60:, 1:2].values  # 前60天作预测基数
    test_set = gru.norm_data(test_set)  
    x_test = []
    x_test.append(test_set[0:, 0])
    predicted_stock_price = gru.predict(x_test, 60, 1)
    yesteday = stock_zh_a_daily_qfq_df.iloc[data_len - 1:, 1:2].values[0]
    print(yesteday)
    print(predicted_stock_price[0])
    
    