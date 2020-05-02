import numpy as np 


def classic_grad_descent(hypothes, max_num_itter, cost_function, alpha=0.01, eps=0.01):
    for _ in range(max_num_itter):
        y_pred = hypothes.hypothesis()
        h_grad = hypothes.hypothesis_grad()
        weight_prev = hypothes.weight.copy()
        loss = cost_function.mean_squared_error(y_pred, hypothes.y)

        hypothes.weight -=  alpha*cost_function.mean_squared_error_grad(y_pred, hypothes.y, h_grad)

        if np.sum(np.abs(weight_prev - hypothes.weight)) < eps:
            break
