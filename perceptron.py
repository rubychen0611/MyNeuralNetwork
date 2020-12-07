import numpy as np
import random
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, init=0.1):
        self.w = np.random.uniform(low=-init, high=init, size=(input_size,))
        self.b = random.uniform(-init, init)

    def train(self, x_train, y_train, epochs=100, lr=0.1):
        for epoch in range(epochs):

            for x, y_true in zip(x_train, y_train):
                s = np.sum(x * self.w) + self.b
                if y_true * s <= 0:
                    self.w += lr * y_true * x
                    self.b += lr * y_true
            y_pred = []
            for xi in x_train:
                y_pred.append(self.predict(xi))
            accuracy = sum(y_train == y_pred) / len(y_train)
            print("epoch=%d, accuracy=%f"% (epoch, accuracy))
            if (epoch+1) % 10 == 0:
                self.visualize(epoch, x_train, y_pred, accuracy)

    def predict(self, x):
        r = np.sum(x * self.w) + self.b
        if r >= 0:
            return 1
        else:
            return -1

