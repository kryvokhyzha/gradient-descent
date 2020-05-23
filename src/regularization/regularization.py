import numpy as np


def get_regularization_func(C, regularization, mini_batch_size=32):
    if regularization is None:
        penalty = lambda x: (x * 0).sum()
        grad_penalty = lambda x: x * 0
    elif regularization == 'L1':
        penalty = lambda x: C*np.abs(x)[1:, :].sum() / (mini_batch_size * 2)
        grad_penalty = lambda x: C*((x >= 0) + (x < 0) * (-1)) / (mini_batch_size * 2)
    elif regularization == 'L2':
        penalty = lambda x: C * np.square(x)[1:, :].sum() / (mini_batch_size * 2)
        grad_penalty = lambda x: C * x / mini_batch_size

    return penalty, grad_penalty