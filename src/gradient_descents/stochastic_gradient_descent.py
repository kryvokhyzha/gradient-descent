import numpy as np
from regularization import get_regularization_func


def stochastic_grad_descent(hypothes, max_num_itter, cost_function, regularization=None, C=1, alpha=0.01, eps=0.01, mini_batch_size=1):
    penalty, grad_penalty = get_regularization_func(C, regularization, len(hypothes.y), mini_batch_size=mini_batch_size)

    weights_history = [hypothes.weight]
    y_pred_history = []
    loss_history = []
    m = len(hypothes.y)

    for _ in range(max_num_itter):
        rand_i = np.random.randint(m, size=(mini_batch_size))
        y_pred = hypothes.hypothesis()
        weight_prev = hypothes.weight.copy()

        y_pred_history.append(y_pred.copy())

        loss = cost_function.get_loss(y_pred, hypothes.y) + penalty(hypothes.weight)
        loss_history.append(loss)

        gp_value = grad_penalty(hypothes.weight)
        gp_value[0, :] = 0

        hypothesis_grad = hypothes.hypothesis_grad()[rand_i]

        hypothes.weight -= alpha * (cost_function.get_grad(y_pred[rand_i], hypothes.y[rand_i], hypothesis_grad) + gp_value)
        weights_history.append(hypothes.weight.copy())

        if (np.abs(weight_prev - hypothes.weight).sum()) < eps:
            print('EPS!')
            break

    return np.array(loss_history), np.array(weights_history), np.array(y_pred_history)
