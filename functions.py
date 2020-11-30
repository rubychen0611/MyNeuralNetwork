import numpy as np
from abc import ABCMeta, abstractmethod

class ActivationFunction:
    '''激活函数基类'''
    __metaclass__ = ABCMeta

    @abstractmethod
    def calculate(self, x):
        pass

    @abstractmethod
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
        return 1 / (1 + np.exp(-x))

    @classmethod
    def derivative(self, x):
        return x * (1 - x)

def mse_loss(y_true, y_pred):
    '''MSE损失函数'''
    return ((y_true - y_pred) ** 2).mean()


