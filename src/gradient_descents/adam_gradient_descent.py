import numpy as np 
from regularization import get_regularization_func


def adam_grad_descent(hypothes, max_num_itter, cost_function, regularization=None, C=1, alpha=0.01, eps=0.01, beta1=0.9, beta2=0.999, epsilon=1e-6, mini_batch_size=32):
    penalty, grad_penalty = get_regularization_func(C, regularization, len(hypothes.y), mini_batch_size=mini_batch_size)
    
    weights_history = [hypothes.weight]
    y_pred_history = []
    loss_history = []
    m = len(hypothes.y)

    avg_gd = np.zeros(hypothes.weight.shape)
    avg_sq_gd = np.zeros(hypothes.weight.shape)

    for i in range(max_num_itter):
        rand_i = np.random.randint(m, size=(mini_batch_size))
        y_pred = hypothes.hypothesis()
        weight_prev = hypothes.weight.copy()

        y_pred_history.append(y_pred.copy())

        loss = cost_function.get_loss(y_pred, hypothes.y) + penalty(hypothes.weight)
        loss_history.append(loss)

        gp_value = grad_penalty(hypothes.weight)
        gp_value[0, :] = 0

        hypothesis_grad = hypothes.hypothesis_grad()[rand_i]
        
        grad = cost_function.get_grad(y_pred[rand_i], hypothes.y[rand_i], hypothesis_grad) + gp_value
        avg_gd = beta1*avg_gd + (1 - beta1)*(grad)
        avg_sq_gd = beta2*avg_sq_gd + (1 - beta2)*(grad)**2

        avg_gd_hat = avg_gd / (1 - np.power(beta1, i+1))
        avg_sq_gd_hat = avg_sq_gd / (1 - np.power(beta2, i+1))

        hypothes.weight -= alpha * ((avg_gd_hat / (np.sqrt(avg_sq_gd_hat) + epsilon)) + gp_value)
        weights_history.append(hypothes.weight.copy())

        if (np.abs(weight_prev - hypothes.weight).sum()) < eps:
            print('EPS!')
            break

    return np.array(loss_history), np.array(weights_history), np.array(y_pred_history)
