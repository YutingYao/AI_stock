# !/bin/python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, GRU
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import datetime
import akshare as ak

class GRU_module(object):
    def __init__(self) -> None:
        super().__init__()
        self.sc = MinMaxScaler(feature_range=(0, 1))
        self.model = tf.keras.Sequential([GRU(80, return_sequences=True), Dropout(0.2), GRU(100), Dropout(0.2), Dense(1)])
        pass
    
    def fit_norm_data(self, data):
        return self.sc.fit_transform(data)
    def norm_data(self, data):
        return self.sc.transform(data)
    def inverse_norm_data(self, data):
        return self.sc.inverse_transform(data)

    def train_module(self, xtrain, ytrain, xtest, ytest, 
        xhh:dict(type=int, info='循环核时间展开步数'), 
        feat_num:dict(type=int, info='每组数据中有多少特征数量')):
        # 打乱训练集合
        np.random.seed(7)
        np.random.shuffle(xtrain)
        np.random.seed(7)
        np.random.shuffle(ytrain)
        tf.random.set_seed(7)
        # 转array格式，方便计算
        xtrain, ytrain = np.array(xtrain), np.array(ytrain)
        '''
        xtrain.shape[0] : 输入数据行数
        60 : 循环时间步数， 输入60天的开盘价格
        1 : 每组数据都是一个特征数量
        '''
        xtrain = np.reshape(xtrain, (xtrain.shape[0], xhh, feat_num))
        # test 数据同理
        xtest, ytest = np.array(xtest), np.array(ytest)
        xtest = np.reshape(xtest, (xtest.shape[0], xhh, feat_num))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error') 

        checkpoint_save_path = "./checkpoint/stock.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            self.model.load_weights(checkpoint_save_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint (filepath=checkpoint_save_path, save_weights_only=True, save_best_only=True, monitor='val_loss')
        
        history = self.model.fit(xtrain, ytrain, batch_size=64, epochs=50, validation_data=(xtest, ytest), validation_freq=1, callbacks=[cp_callback])
        self.model.summary()
        file = open('./weights.txt', 'w')  # 参数提取
        for v in self.model.trainable_variables:
            file.write(str(v.name) + '\n')
            file.write(str(v.shape) + '\n')
            file.write(str(v.numpy()) + '\n')
        file.close()
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()
        pass

    def predict(self, 
        xtest:dict(type=list, info='待预测数据特征'), 
        xhh:dict(type=int, info='循环核时间展开步数'), 
        feat_num:dict(type=int, info='每组数据中有多少特征数量')):
        xtest = np.array(xtest)
        xtest = np.reshape(xtest, (xtest.shape[0], xhh, feat_num))
        # 测试集输入模型进行预测
        predicted_stock_price = self.model.predict(xtest)
        # 对预测数据还原---从（0，1）反归一化到原始范围
        return self.sc.inverse_transform(predicted_stock_price)