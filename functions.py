import math

import numpy as np
from abc import ABCMeta, abstractmethod

class ActivationFunction:
    '''激活函数基类'''
    __metaclass__ = ABCMeta

    @abstractmethod
    def calculate(self, x):
        pass

    def derivative(self, x):
        pass

class ReLU(ActivationFunction):
    '''relu激活函数'''

    @classmethod
    def calculate(self, x):
        return np.maximum(x, 0)

    @classmethod
    def derivative(self, x):
        return np.where(x > 0, 1, 0)

class Pureline(ActivationFunction):
    '''恒等激活函数'''

    @classmethod
    def calculate(self, x):
        return x

    @classmethod
    def derivative(self, x):
        return np.ones(shape=x.shape)

class Sigmoid(ActivationFunction):

    @classmethod
    def calculate(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    @classmethod
    def derivative(self, x):
        s = self.calculate(x)
        return s * (1.0 - s)

class Softmax(ActivationFunction):
    @classmethod
    def calculate(self, x):
        shift_x = x - np.max(x)  # 防止输入增大时输出为nan
        exp_x = np.exp(shift_x)
        return exp_x / np.sum(exp_x)


class LossFunction:
    '''激活函数基类'''
    __metaclass__ = ABCMeta

    @abstractmethod
    def calculate(self, y_true, y_pred):
        pass

    def derivative(self, y_true, y_pred):
        pass


class MSE(LossFunction):
    '''MSE损失函数'''
    @classmethod
    def calculate(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()
    @classmethod
    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / len(y_pred)

class CrossEntropy(LossFunction):
    '''交叉熵损失函数'''
    @classmethod
    def calculate(self, y_true, y_pred):
        return np.sum(np.nan_to_num(-y_true*np.log(y_pred)-(1-y_true)*np.log(1-y_pred)))

