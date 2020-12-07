import random
import shutil

import numpy as np
import os
import pickle
from functions import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

figure_dir = './figure'
model_dir = './model'
'''多隐层神经网络'''
class NeuralNetwork:
    __metaclass__ = ABCMeta
    def __init__(self, input_size, output_size, hidden_size, model_path=None, output_activated='sigmoid', loss_function='mse', init=0.5):
        '''
        初始化
        :param input_size: 输入层大小
        :param output_size: 输出层大小
        :param hidden_size: 数组表示隐藏层个数以及每层神经元个数
        '''

        self.len = len(hidden_size) + 1    # 层数(除输入层)
        if model_path is not None:  # 从文件中加载权重
            f = open(model_path, 'rb')
            self.w, self.b = pickle.load(f)
            f.close()
            self.activations = [ReLU] * len(hidden_size)
            if output_activated == 'sigmoid':
                self.activations.append(Sigmoid)
            elif output_activated == 'relu':
                self.activations.append(ReLU)
            elif output_activated == 'pureline':
                self.activations.append(Pureline)
            elif output_activated == 'softmax':
                self.activations.append(Softmax)
        else:
            # 随机初始化权重和偏移量
            # 隐藏层
            self.w = [self.random_initialize(shape=(input_size, hidden_size[0]), alpha=init)]
            self.b = [self.random_initialize(shape=(hidden_size[0],), alpha=init)]
            self.activations = [ReLU]
            for i in range(len(hidden_size)-1):
                self.w.append(self.random_initialize(shape=(hidden_size[i], hidden_size[i+1]), alpha=init))
                self.b.append(self.random_initialize(shape=(hidden_size[i+1],), alpha=init))
                self.activations.append(ReLU)
            # 输出层
            self.w.append(self.random_initialize(shape=(hidden_size[-1], output_size), alpha=init))
            self.b.append(self.random_initialize(shape=(output_size,), alpha=init))
            if output_activated == 'sigmoid':
                self.activations.append(Sigmoid)
            elif output_activated == 'relu':
                self.activations.append(ReLU)
            elif output_activated == 'pureline':
                self.activations.append(Pureline)
            elif output_activated == 'softmax':
                self.activations.append(Softmax)
        # 损失函数
        if loss_function == 'mse':
            self.loss_function = MSE
        elif loss_function == 'cross_entropy':
            self.loss_function = CrossEntropy
        # 初始化中间结果数组a、z、delta、delta_w、delta_b
        self.z = []
        self.a = []
        self.delta = []
        self.delta_b = []
        for bi in self.b:
            self.z.append(np.zeros(shape=bi.shape))
            self.a.append(np.zeros(shape=bi.shape))
            self.delta.append(np.zeros(shape=bi.shape))
            self.delta_b.append(np.zeros(shape=bi.shape))
        self.delta_w = []
        for wi in self.w:
            self.delta_w.append(np.zeros(shape=wi.shape))
        # 初始化中间结果文件夹
        if os.path.exists(figure_dir):
            shutil.rmtree(figure_dir)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.mkdir(figure_dir)
        os.mkdir(model_dir)

    def random_initialize(self, shape, alpha=0.5):
        x = np.random.uniform(low=-alpha, high=alpha, size=shape)
        x *= alpha
        return x

    def feedforward(self, x):
        '''前向传播计算预测结果'''
        layer_output = x
        for layer in range(0, self.len):
            layer_output = self.activations[layer].calculate((np.matmul(layer_output, self.w[layer]) + self.b[layer]))
        return layer_output.squeeze()

    def cal_train_loss(self, x_train, y_train):
        '''计算训练集和验证集上的损失，生成可视化图像'''
        y_pred_train = self.feedforward(x_train)
        train_loss = self.loss_function.calculate(y_train, y_pred_train)
        return train_loss

    def cal_val_loss(self, x_val, y_val):
        y_pred_val = self.feedforward(x_val)
        val_loss = self.loss_function.calculate(y_val, y_pred_val)
        return val_loss

    def test(self, epoch, x_train, y_train, x_val, y_val):
        train_loss = self.cal_train_loss(x_train, y_train)
        if x_val is not None:
            val_loss = self.cal_val_loss(x_val, y_val)
            print("%d\t%.5f\t%.5f" % (epoch, train_loss, val_loss))
        else:
            val_loss = -1
            print("%d\t%.5f" % (epoch, train_loss))
        self.visualize(epoch, x_train, y_train, train_loss, val_loss)

    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=30, batch_size=10, lr=0.01):
        '''训练模型'''
        self.test(-1, x_train, y_train, x_val, y_val)

        for epoch in range(epochs):
            x_train_shuffle, y_train_shuffle = self.shuffle(x_train, y_train)
            x_train_shuffle = x_train
            y_train_shuffle = y_train
            batch_count = 0
            for x, y_true in zip(x_train_shuffle, y_train_shuffle):
                # 前向传播得到每层输出
                layer_output = x
                for layer in range(0, self.len):
                    self.z[layer] = np.matmul(layer_output, self.w[layer]) + self.b[layer]
                    layer_output = self.activations[layer].calculate(self.z[layer])
                    self.a[layer] = layer_output

                # 计算最后一层的神经单元误差δ
                if self.loss_function == CrossEntropy:      # 分类问题：使用softmax+交叉熵损失函数时
                    self.delta[-1] = self.a[-1] - y_true
                else:
                    self.delta[-1] = self.loss_function.derivative(y_true, self.a[-1]) * self.activations[-1].derivative(self.z[-1])
                # 反向递推计算每层神经单元误差δ
                for layer in range(self.len-2, -1, -1):
                    self.delta[layer] = np.matmul(self.w[layer + 1], self.delta[layer + 1]) * self.activations[layer].derivative(self.z[layer])

                # 累加导数
                last_layer_output = x
                for layer in range(self.len):
                    self.delta_w[layer] += np.matmul(np.transpose([last_layer_output]), [self.delta[layer]])
                    self.delta_b[layer] += self.delta[layer]
                    last_layer_output = self.a[layer]

                # 每个batch结束后更新w、b
                batch_count += 1
                if batch_count == batch_size:
                    for layer in range(self.len):
                        self.w[layer] -= (lr * self.delta_w[layer] / batch_size)
                        self.b[layer] -= (lr * self.delta_b[layer] / batch_size)
                        self.delta_w[layer].fill(0)
                        self.delta_b[layer].fill(0)
                    batch_count = 0
            self.test(epoch, x_train, y_train, x_val, y_val)
            if epoch > 0 and epoch % 1000 == 0:
                lr *= 0.98

    def save_model(self, epoch):
        '''保存模型'''
        f = open("model/model_epoch=%04d.pkl" % epoch, 'wb')
        pickle.dump([self.w, self.b], f)
        f.close()

    def shuffle(self, x, y):
        '''随机打乱数据'''
        d = 1
        if y.ndim > 1:
            d = y.shape[1]
        xy= np.c_[x,y]
        np.random.shuffle(xy)
        x = xy[:, :-d]
        y = xy[:, -d:]
        return x, y

    @abstractmethod
    def visualize(self, epoch,  x_train, y_train, train_loss, val_loss):
        pass
