import numpy as np 
import streamlit as st
from sklearn.metrics import mean_squared_error


def classic_grad_descent(hypothes, max_num_itter, cost_function, regularization=None, C=1, alpha=0.01, eps=0.01):
    if regularization is None:
        penalty = lambda x: (x * 0).sum()
        grad_penalty = lambda x: x * 0
        print('None')
    elif regularization == 'L1':
        penalty = lambda x: C*np.abs(x)[:, 1:].sum() / len(hypothes.y)
        grad_penalty = lambda x: C*((x > 0) + (x < 0) * (-1))
        print('L1')
    elif regularization == 'L2':
        penalty = lambda x: C * np.square(x)[:, 1:].sum() / (len(hypothes.y)*2)
        grad_penalty = lambda x: C * x / len(hypothes.y)
        print('L2')

    I = np.eye(hypothes.X.shape[1])
    I[0, :] = 0
    st.text('BEST')
    q1 = hypothes.X.T @ hypothes.y
    q2 = np.linalg.pinv(hypothes.X.T @ hypothes.X + C*I)
    st.write(q2 @ q1)

    weights_history = [hypothes.weight]
    loss_history = []

    for i in range(max_num_itter):
        y_pred = hypothes.hypothesis()
        weight_prev = hypothes.weight.copy()

        loss = cost_function.mean_squared_error(y_pred, hypothes.y) + penalty(hypothes.weight)
        loss_history.append(loss)

        gp_value = grad_penalty(hypothes.weight)
        gp_value[0, :] = 0
        
        hypothes.weight -= alpha * (cost_function.mean_squared_error_grad(y_pred, hypothes.y, hypothes.hypothesis_grad()) + gp_value)

        weights_history.append(hypothes.weight)

        if (np.abs(weight_prev - hypothes.weight).sum()) < eps:
            print('EPS!')
            break

    return loss_history, weights_history
