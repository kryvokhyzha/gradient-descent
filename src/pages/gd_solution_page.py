import streamlit as st
import pandas as pd
import numpy as np
import os
import time

from collections import namedtuple
from sklearn.datasets import make_regression, make_classification

import matplotlib.pyplot as plt

from utils.constants import *
from db import db_insert
from plot import plot_all



def show_side_bar():
    st.sidebar.header('Algorithm modification')
    modification = st.sidebar.selectbox('', key='modification_slbox',
                                        options=list(MODIFICATIONS.keys()))

    st.sidebar.header('Hypothesis function')
    hypothesis = st.sidebar.selectbox('', key='hypothesis_slbox',
                                      options=list(HYPOTHESES.keys()))

    if hypothesis == 'Polynomial':
        st.sidebar.header('Polynomial degree')
        degree = int(st.sidebar.number_input('', key='degree', min_value=1, value=2, step=1))
    else:
        degree = 1

    st.sidebar.header('Cost function')
    cost_function = st.sidebar.selectbox('', key='costf_slbox',
                                         options=list(COST_FUNCTIONS.keys()))

    st.sidebar.header('Regularization')
    regularization = st.sidebar.selectbox('', key='regularization_slbox',
                                          options=list(REGULARIZATION.keys()))

    st.sidebar.header('Scaling function')
    scaler = st.sidebar.selectbox('', key='scale_slbox', options=list(SCALE.keys()))

    if regularization != 'None':
        st.sidebar.header('Regularization coeff')
        reg_coef = float(st.sidebar.number_input('', key='reg_coef', min_value=0.0, value=1.0, step=0.1))
    else:
        reg_coef = 0.0

    st.sidebar.header('Learning rate')
    alpha = st.sidebar.slider('', 0.001, 0.1, step=0.001, format='%f', key='learning_rate')

    st.sidebar.header('Early stopping')
    eps = st.sidebar.slider('', 0.0, 0.1, step=0.001, format='%f', key='early_stopping')

    st.sidebar.header('Max number of itteration')
    max_num_itter = int(st.sidebar.number_input('', key='max_num_itter', min_value=1, value=100, step=1))

    Properties = namedtuple('Properties', ['modification', 'hypothesis', 'degree', 'cost_function',
                            'scaler', 'regularization', 'reg_coef', 'alpha', 'eps', 'max_num_itter'])
    
    Choice = namedtuple('Choice', ['modification', 'hypothesis', 'cost_function',
                            'scaler', 'regularization'])

    return Properties(modification=MODIFICATIONS[modification], hypothesis=HYPOTHESES[hypothesis], degree=degree,
                      cost_function=COST_FUNCTIONS[cost_function], scaler=SCALE[scaler],
                      regularization=REGULARIZATION[regularization], reg_coef=reg_coef,
                      eps=eps, alpha=alpha, max_num_itter=max_num_itter), Choice(modification=modification,
                       hypothesis=hypothesis, cost_function=cost_function, scaler=scaler, regularization = regularization)
            


def select_task_type():
    st.header('Please, select type task')
    task_type = st.selectbox('', key='type_slbox', options=['Individual', 'Generate regression task', 'Generate classification task'])

    return task_type


def params_for_generate_regression():
    st.header('Please, select parameters for dataset generation')

    n_samples = int(st.number_input('The number of samples', key='n_samples_r', min_value=1, value=100, step=1))

    n_features = int(st.number_input('The number of features', key='n_features_r', min_value=1, value=1, step=1))

    n_informative = int(st.number_input('The number of informative features', key='n_informative_r', min_value=1, max_value=n_features, value=1, step=1))

    degree = int(st.number_input('The number of degree', key='degree_r', min_value=1, value=1, step=1))

    noise = float(st.number_input('The standard deviation of the gaussian noise applied to the output',
                                          key='noise_r', min_value=0.0, value=10.0, step=0.1))

    return {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_informative': n_informative,
        'noise': noise
    }, degree


def params_for_generate_classification():
    st.header('Please, select parameters for dataset generation')

    n_samples = int(st.number_input('The number of samples', key='n_samples_c', min_value=1, value=100, step=1))

    n_features = int(st.number_input('The number of features', key='n_features_c', min_value=1, value=1, step=1))

    n_informative = int(st.number_input('The number of informative features', key='n_informative_c', min_value=1, max_value=n_features, value=1, step=1))

    n_redundant_title = 'The number of redundant features. These features are generated as random linear combinations of the informative features'
    n_redundant = int(st.number_input(n_redundant_title, key='n_redundant_c', min_value=0, max_value=int(n_features - n_informative), value=0, step=1))

    n_clusters_per_class = int(st.number_input('The number of clusters per class.', key='n_clusters_per_class_c', min_value=1, max_value=n_informative, value=1, step=1))

    return {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_redundant': n_redundant,
        'n_informative': n_informative,
        'n_clusters_per_class': n_clusters_per_class
    }


def generate_regression_task(h_type, degree, scaler, data_degree, **kwargs):
    X, y = make_regression(**kwargs)
    y = y.reshape((len(y), 1))
    if scaler is not None:
        X = scaler.fit_transform(X)
        y = scaler.fit_transform(y)

    y = y ** data_degree
    return h_type(X, y, degree=degree)


def generate_clasiffication_task(h_type, degree, scaler, **kwargs):
    X, y = make_classification(**kwargs)
    if scaler is not None:
        X = scaler.fit_transform(X)

    y = y.reshape((len(y), 1))
    return h_type(X, y, degree=degree)


def individual_task(h_type, degree, scaler):
    df = pd.read_csv('data/restaurant_revenue.txt', header=None, sep=',')
    if scaler is not None:
        X = scaler.fit_transform(df[[0]].values)
        y = scaler.fit_transform(df[[1]].values)
    else:
        X = df[[0]].values
        y = df[[1]].values
    return h_type(X, y, degree=degree)


def solve_btn(h, properties, choice):
    if st.button('Solve', key='solve_btn'):
        start_time = time.time()
        st.write(h.weight)
        with st.spinner('waiting...'):
            loss_history, weights_history, y_pred_history = properties.modification(h, properties.max_num_itter, properties.cost_function,
                                                                                    regularization=properties.regularization, C=properties.reg_coef,
                                                                                    alpha=properties.alpha, eps=properties.eps)
            st.success('Finished!')

        if np.isnan(h.weight).any():
            st.error("Result approximates to infinity. Please, select another parameters.")
        else:
            st.write(h.weight)
        
        db_insert(h, properties, time.time() - start_time, choice)

        plot_all(h, properties, weights_history, loss_history, y_pred_history)
        

def gd_solution_page():
    st.title('Gradient Descent')
    properties, choice = show_side_bar()
    task_type = select_task_type()
    
    if task_type == 'Individual':
        h = individual_task(properties.hypothesis, properties.degree, properties.scaler)
    elif task_type == 'Generate regression task':
        kwargs, degree = params_for_generate_regression()
        h = generate_regression_task(properties.hypothesis, properties.degree, properties.scaler, degree, **kwargs)
    elif task_type == 'Generate classification task':
        kwargs = params_for_generate_classification()
        h = generate_clasiffication_task(properties.hypothesis, properties.degree, properties.scaler, **kwargs)
    
    solve_btn(h, properties, choice)
