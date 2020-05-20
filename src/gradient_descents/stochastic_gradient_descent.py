import numpy as np 
import streamlit as st
from sklearn.metrics import mean_squared_error 

def stochastic_grad_descent(hypothes, max_num_itter, cost_function, regularization=None, C=1, alpha=0.01, eps=0.01):
    if regularization is None:
        penalty = lambda x: (x * 0).sum()
        grad_penalty = lambda x: x * 0
    elif regularization == 'L1':
        penalty = lambda x: C*np.abs(x)[:, 1:].sum() / len(hypothes.y)
        grad_penalty = lambda x: C*((x > 0) + (x < 0) * (-1))
    elif regularization == 'L2':
        penalty = lambda x: C * np.square(x)[:, 1:].sum() / (len(hypothes.y)*2)
        grad_penalty = lambda x: C * x / len(hypothes.y)

    I = np.eye(hypothes.X.shape[1])
    I[0, :] = 0
    st.text('BEST')
    q1 = hypothes.X.T @ hypothes.y
    q2 = np.linalg.pinv(hypothes.X.T @ hypothes.X + C*I)
    st.write(q2 @ q1)

    weights_history = [hypothes.weight]
    y_pred_history = []
    loss_history = []
    m = len(hypothes.y)

    for _ in range(max_num_itter):
        loss = 0
        for _ in range(m):
            rand_i = np.random.randint(0,m)
            y_pred = hypothes.hypothesis()
            weight_prev = hypothes.weight.copy()

            loss += cost_function.get_loss(y_pred, hypothes.y) + penalty(hypothes.weight)
            
            gp_value = grad_penalty(hypothes.weight)
            gp_value[0, :] = 0

            hypothesis_grad = hypothes.hypothesis_grad()
            Xi = hypothesis_grad[rand_i,:].reshape(1,hypothesis_grad.shape[1])
            yi = hypothes.y[rand_i].reshape(1,1)
            pred = y_pred[rand_i].reshape(1,1)

            hypothes.weight -= alpha * (cost_function.get_grad(pred, yi, Xi) + gp_value)

            if (np.abs(weight_prev - hypothes.weight).sum()) < eps:
                print('EPS!')
                break
        loss_history.append(loss)
        weights_history.append(hypothes.weight.copy())
        y_pred_history.append(y_pred.copy())

    return loss_history, np.array(weights_history), np.array(y_pred_history)
