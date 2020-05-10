import numpy as np


class MSE:
    @staticmethod
    def mean_squared_error_grad(y_pred, y, h_grad):
        """
        h_grad - gradient of hypothesis function
        """
        return 2 * (y_pred - y).T @ h_grad / len(y_pred)

    @staticmethod
    def mean_squared_error(y_pred, y):
        return np.square(y_pred - y).mean() 
        