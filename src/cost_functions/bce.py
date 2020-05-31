import numpy as np


class BCE:
    @staticmethod
    def get_grad(y_pred, y, h_grad):
        return h_grad.T.dot(y_pred - y) / (2*len(y))

    @staticmethod
    def get_loss(y_pred, y, eps=1e-6):
        return -np.sum(y*np.log(y_pred + eps) + (1 - y)*np.log(1 - y_pred + eps)) / (2*len(y))