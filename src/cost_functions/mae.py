import numpy as np 


class MAE:
    @staticmethod
    def get_grad(y_pred, y, h_grad):
        return h_grad.T.dot(np.sign(y_pred - y)) / (2*len(y))

    @staticmethod
    def get_loss(y_pred, y):
        return np.sum(np.abs(y_pred - y)) / (2*len(y))
