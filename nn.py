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
    def __init__(self, input_size, output_size, hidden_size, model_path=None):
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
            self.activations.append(Pureline)
        else:
            # 随机初始化权重和偏移量
            # 隐藏层
            loc = 0.0
            scale = 0.1
            self.w = [np.random.normal(loc, scale, size=(input_size, hidden_size[0]))]
            self.b = [np.random.normal(loc, scale, size=(hidden_size[0],))]
            self.activations = [ReLU]
            for i in range(len(hidden_size)-1):
                self.w.append(np.random.normal(loc, scale, size=(hidden_size[i], hidden_size[i+1])))
                self.b.append(np.random.normal(loc, scale, size=(hidden_size[i+1],)))
                self.activations.append(ReLU)
            # 输出层
            self.w.append(np.random.normal(loc, scale, size=(hidden_size[-1], output_size)))
            self.b.append(np.random.normal(loc, scale, size=(output_size,)))
            self.activations.append(Pureline)
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


    def feedforward(self, x):
        '''前向传播计算预测结果'''
        layer_output = x
        for layer in range(0, self.len):
            layer_output = self.activations[layer].calculate((np.matmul(layer_output, self.w[layer]) + self.b[layer]))
        return layer_output.squeeze()

    def validate(self, epoch, x_train, y_train, x_val, y_val):
        '''计算训练集和验证集上的损失，生成可视化图像'''
        y_pred_train = self.feedforward(x_train)
        train_loss = mse_loss(y_train, y_pred_train)
        y_pred_val = self.feedforward(x_val)
        val_loss = mse_loss(y_val, y_pred_val)
        self.visualize(epoch, y_pred_train, train_loss, val_loss)
        #print("Epoch %d, train loss: %.4f, validation loss: %.4f" % (epoch, train_loss, val_loss))
        print("%d\t%.5f\t%.5f" % (epoch, train_loss, val_loss))

    def train(self, x_train, y_train, x_val, y_val, epochs=30, batch_size=10, lr=0.01):
        '''训练模型'''

        self.validate(0, x_train, y_train, x_val, y_val)

        for epoch in range(epochs):
            x_train_shuffle, y_train_shuffle = shuffle(x_train, y_train)
            batch_count = 0
            for x, y_true in zip(x_train_shuffle, y_train_shuffle):
                # 前向传播得到每层输出
                layer_output = x
                for layer in range(0, self.len):
                    self.z[layer] = np.matmul(layer_output, self.w[layer]) + self.b[layer]
                    layer_output = self.activations[layer].calculate(self.z[layer])
                    self.a[layer] = layer_output

                # 计算最后一层的神经单元误差δ
                self.delta[-1] = 2 * (self.a[-1] - y_true) * self.activations[-1].derivative(self.z[-1]) / len(self.delta[-1])

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
                        self.w[layer] -= lr * self.delta_w[layer]
                        self.b[layer] -= lr * self.delta_b[layer]
                        self.delta_w[layer].fill(0)
                        self.delta_b[layer].fill(0)
                    batch_count = 0

            self.validate(epoch + 1, x_train, y_train, x_val, y_val)

            if (epoch + 1) % 3 == 0:
                lr *= 0.9
            if (epoch + 1) % 10 == 0:
                # 保存模型
                self.save_model(epoch + 1)

    def save_model(self, epoch):
        '''保存模型'''
        f = open("model/model_epoch=%04d.pkl" % epoch, 'wb')
        pickle.dump([self.w, self.b], f)
        f.close()

    def visualize(self, epoch, y_pred, train_loss, val_loss):
        '''可视化训练结果'''
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(-5, 5, 0.1)
        Y = np.arange(-5, 5, 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = y_pred.reshape((100,100))
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
        ax.set_zlim(-5, 5)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_title('epoch=%d, train loss=%.4f, validation loss=%.4f'% (epoch, train_loss, val_loss))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('figure/%04d.png' % epoch)
        plt.show()


def shuffle(x, y):
    '''随机打乱数据'''
    xy= np.c_[x,y]
    np.random.shuffle(xy)
    x = xy[:, :-1]
    y = xy[:, -1]
    return x, y

def gen_dataset():
    # 生成训练数据集
    x_train = []
    y_train = []
    for x1 in np.arange(-5, 5, 0.1):
        for x2 in np.arange(-5, 5, 0.1):
            x_train.append([x1, x2])
            y_train.append(np.sin(x1)-np.cos(x2))
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # 生成验证数据集
    x_val = []
    y_val = []
    for i in range(1000):
        x1 = random.uniform(-5, 5)
        x2 = random.uniform(-5, 5)
        x_val.append([x1, x2])
        y_val.append(np.sin(x1)-np.cos(x2))
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    return x_train, y_train, x_val, y_val



x_train, y_train, x_val, y_val = gen_dataset()
nn = NeuralNetwork(input_size=2, output_size=1, hidden_size=[100,80,50,30,10])
nn.train(x_train, y_train, x_val, y_val)