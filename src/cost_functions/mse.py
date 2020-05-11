import numpy as np


class MSE:
    @staticmethod
    def mean_squared_error_grad(y_pred, y, h_grad):
        """
        h_grad - gradient of hypothesis function
        """
        return h_grad.T.dot(y_pred - y) / (len(y))

    @staticmethod
    def mean_squared_error(y_pred, y):
        return np.sum(np.square(y_pred - y)) / (2*len(y))
        