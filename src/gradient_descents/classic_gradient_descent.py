import numpy as np 
from regularization import get_regularization_func


def classic_grad_descent(hypothes, max_num_itter, cost_function, regularization=None, C=1, alpha=0.01, eps=0.01, mini_batch_size=32):
    penalty, grad_penalty = get_regularization_func(C, regularization, len(hypothes.y), mini_batch_size=len(hypothes.y))

    weights_history = [hypothes.weight.copy()]
    y_pred_history = []
    loss_history = []

    for _ in range(max_num_itter):
        y_pred = hypothes.hypothesis()
        weight_prev = hypothes.weight.copy()

        y_pred_history.append(y_pred.copy())

        loss = cost_function.get_loss(y_pred, hypothes.y) + penalty(hypothes.weight)
        loss_history.append(loss)

        gp_value = grad_penalty(hypothes.weight)
        gp_value[0, :] = 0
        
        hypothes.weight -= alpha * (cost_function.get_grad(y_pred, hypothes.y, hypothes.hypothesis_grad()) + gp_value)

        weights_history.append(hypothes.weight.copy())

        if (np.abs(weight_prev - hypothes.weight).sum()) < eps:
            print('EPS!')
            break

    return np.array(loss_history), np.array(weights_history), np.array(y_pred_history)
