import numpy as np
from regularization import get_regularization_func


def compute_j_grid(h, theta0_grid, theta1_grid, cost_function, C=1, regularization=None):
    penalty, _ = get_regularization_func(C, regularization, len(h.y), mini_batch_size=len(h.y))

    grid = []
    for theta1 in theta1_grid:
        row = []
        for theta0 in theta0_grid:
            w = np.array([[theta0], [theta1]])
            y_pred = h.hypothesis(w=w)
            elem = cost_function.get_loss(y_pred, h.y) + penalty(w)
            row.append(elem)
        grid.append(row)
    return np.array(grid)


def compute_j(h, theta0, theta1, cost_function, C=1, regularization=None):
    assert len(theta0) == len(theta1)

    penalty, _ = get_regularization_func(C, regularization, len(h.y), mini_batch_size=len(h.y))

    loss = []
    for i in range(len(theta0)):
        w = np.array([[theta0[i]], [theta1[i]]])
        y_pred = h.hypothesis(w=w)
        elem = cost_function.get_loss(y_pred, h.y) + penalty(w)
        loss.append(elem)
    return loss
