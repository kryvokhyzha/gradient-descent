import numpy as np 
import streamlit as st


def classic_grad_descent(hypothes, max_num_itter, cost_function, alpha=0.01, eps=0.001):
    for _ in range(max_num_itter):
        y_pred = hypothes.hypothesis()
        h_grad = hypothes.hypothesis_grad()
        weight_prev = hypothes.weight.copy()
        loss = cost_function.mean_squared_error(y_pred, hypothes.y)

        hypothes.weight -=  alpha*cost_function.mean_squared_error_grad(y_pred, hypothes.y, h_grad)

        if ((np.abs(weight_prev - hypothes.weight).sum(axis=1)) < eps).all():
            print('EPS!!!!!!!!!!!!!!!!!!!')
            break
