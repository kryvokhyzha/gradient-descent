import numpy as np


class MSE:
    @staticmethod
    def get_grad(y_pred, y, h_grad):
        return h_grad.T.dot(y_pred - y) / (len(y))

    @staticmethod
    def get_loss(y_pred, y):
        return np.sum(np.square(y_pred - y)) / (2*len(y))
        