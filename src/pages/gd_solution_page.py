import streamlit as st
import pandas as pd
import os
from collections import namedtuple

from gradient_descents import classic_grad_descent
from hypotheses import Linear
from cost_functions import MSE


MODIFICATIONS = {
    'Classic GD': classic_grad_descent,
    'SGD': classic_grad_descent,
    'SGD with Momentum': classic_grad_descent,
    'RMSProp': classic_grad_descent,
    'Adam': classic_grad_descent
}


HYPOTHESES = {
    'Linear': Linear,
    'Square': Linear
}


COST_FUNCTIONS = {
    'MSE': MSE,
    'MAE': MSE
}


def show_side_bar():
    st.sidebar.header('Algorithm modification')
    modification = st.sidebar.selectbox('', key='modification_slbox',
                                        options=['Classic GD', 'SGD', 'SGD with Momentum',
                                                 'RMSProp', 'Adam'])

    st.sidebar.header('Hypothesis function')
    hypothesis = st.sidebar.selectbox('', key='hypothesis_slbox',
                                      options=['Linear', 'Square'])

    st.sidebar.header('Cost function')
    cost_function = st.sidebar.selectbox('', key='costf_slbox',
                                         options=['MSE', 'MAE'])

    st.sidebar.header('Regularization')
    regularization = st.sidebar.selectbox('', key='regularization_slbox',
                                          options=['None', 'L1', 'L2'])

    st.sidebar.header('Learning rate')
    alpha = st.sidebar.slider('', 0.001, 0.1, step=0.001, format='%f', key='learning_rate')

    st.sidebar.header('Early stopping')
    eps = st.sidebar.slider('', 0.001, 0.1, step=0.001, format='%f', key='early_stopping')

    st.sidebar.header('Max number of itteration')
    max_num_itter = int(st.sidebar.number_input('', key='max_num_itter', min_value=1, value=100, step=1))

    Properties = namedtuple('Properties', ['modification', 'hypothesis', 'cost_function', 'regularization', 'alpha', 'eps', 'max_num_itter'])

    return Properties(modification=MODIFICATIONS[modification], hypothesis=HYPOTHESES[hypothesis],
                      cost_function=COST_FUNCTIONS[cost_function], regularization=regularization,
                      eps=eps, alpha=alpha, max_num_itter=max_num_itter)


def select_task_type():
    st.header('Please, select type task')
    task_type = st.selectbox('', key='type_slbox', options=['Individual', 'Generate task'])

    return task_type


def generate_task(h_type):
    pass


def individual_task(h_type):
    df = pd.read_csv('data/restaurant_revenue.txt', header=None, sep=',')
    return h_type(df[[0]], df[[1]])


def gd_solution_page():
    st.title('Gradient Descent')
    properties = show_side_bar()
    task_type = select_task_type()

    st.write([properties])

    if task_type == 'Individual':
        h = individual_task(properties.hypothesis)
    elif task_type == 'Generate task':
        generate_task(properties.hypothesis)

    if st.button('Solve', key='solve_btn'):
        st.write(h.weight)
        classic_grad_descent(h, properties.max_num_itter, properties.cost_function, alpha=properties.alpha, eps=properties.eps)
        st.write(h.weight)
