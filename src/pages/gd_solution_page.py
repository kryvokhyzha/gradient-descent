import streamlit as st
import pandas as pd
import os

from collections import namedtuple
from sklearn.datasets import make_regression, make_classification

from utils.constants import *
from plot import cost_function_plot_2d


def show_side_bar():
    st.sidebar.header('Algorithm modification')
    modification = st.sidebar.selectbox('', key='modification_slbox',
                                        options=list(MODIFICATIONS.keys()))

    st.sidebar.header('Hypothesis function')
    hypothesis = st.sidebar.selectbox('', key='hypothesis_slbox',
                                      options=list(HYPOTHESES.keys()))

    st.sidebar.header('Cost function')
    cost_function = st.sidebar.selectbox('', key='costf_slbox',
                                         options=list(COST_FUNCTIONS.keys()))

    st.sidebar.header('Regularization')
    regularization = st.sidebar.selectbox('', key='regularization_slbox',
                                          options=list(REGULARIZATION.keys()))

    st.sidebar.header('Scaling function')
    scaler = st.sidebar.selectbox('', key='scale_slbox', options=list(SCALE.keys()))

    st.sidebar.header('Regularization coeff')
    reg_coef = float(st.sidebar.number_input('', key='reg_coef', min_value=0.0, value=1.0, step=0.1))

    st.sidebar.header('Learning rate')
    alpha = st.sidebar.slider('', 0.001, 0.1, step=0.001, format='%f', key='learning_rate')

    st.sidebar.header('Early stopping')
    eps = st.sidebar.slider('', 0.0, 0.1, step=0.001, format='%f', key='early_stopping')

    st.sidebar.header('Max number of itteration')
    max_num_itter = int(st.sidebar.number_input('', key='max_num_itter', min_value=1, value=100, step=1))

    Properties = namedtuple('Properties', ['modification', 'hypothesis', 'cost_function',
                            'scaler', 'regularization', 'reg_coef', 'alpha', 'eps', 'max_num_itter'])

    return Properties(modification=MODIFICATIONS[modification], hypothesis=HYPOTHESES[hypothesis],
                      cost_function=COST_FUNCTIONS[cost_function], scaler=SCALE[scaler],
                      regularization=REGULARIZATION[regularization], reg_coef=reg_coef,
                      eps=eps, alpha=alpha, max_num_itter=max_num_itter)


def select_task_type():
    st.header('Please, select type task')
    task_type = st.selectbox('', key='type_slbox', options=['Individual', 'Generate regression task', 'Generate classification task'])

    return task_type


def params_for_generate_regression():
    st.header('Please, select parameters for dataset generation')

    n_samples = int(st.number_input('The number of samples', key='n_samples_r', min_value=1, value=100, step=1))

    n_features = int(st.number_input('The number of features', key='n_features_r', min_value=1, value=1, step=1))

    n_informative = int(st.number_input('The number of informative features', key='n_informative_r', min_value=1, value=1, step=1))

    noise = float(st.number_input('The standard deviation of the gaussian noise applied to the output',
                                          key='noise_r', min_value=0.0, value=10.0, step=0.1))

    return {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_informative': n_informative,
        'noise': noise
    }


def params_for_generate_classification():
    st.header('Please, select parameters for dataset generation')

    n_samples = int(st.number_input('The number of samples', key='n_samples_c', min_value=1, value=100, step=1))

    n_features = int(st.number_input('The number of features', key='n_features_c', min_value=1, value=1, step=1))

    n_redundant_title = 'The number of redundant features. These features are generated as random linear combinations of the informative features'
    n_redundant = int(st.number_input(n_redundant_title, key='n_redundant_c', min_value=0, value=0, step=1))

    n_informative = int(st.number_input('The number of informative features', key='n_informative_c', min_value=1, value=1, step=1))

    n_clusters_per_class = int(st.number_input('The number of clusters per class.', key='n_clusters_per_class_c', min_value=1, value=1, step=1))

    return {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_redundant': n_redundant,
        'n_informative': n_informative,
        'n_clusters_per_class': n_clusters_per_class
    }


def generate_regression_task(h_type, scaler, **kwargs):
    X, y = make_regression(**kwargs)
    if scaler is not None:
        X = scaler.fit_transform(X)

    y = y.reshape((len(y), 1))
    return h_type(X, y)


def generate_clasiffication_task(h_type, scaler, **kwargs):
    X, y = make_classification(**kwargs)
    if scaler is not None:
        X = scaler.fit_transform(X)

    y = y.reshape((len(y), 1))
    return h_type(X, y)


def individual_task(h_type, scaler):
    df = pd.read_csv('data/restaurant_revenue.txt', header=None, sep=',')
    if scaler is not None:
        X = scaler.fit_transform(df[[0]].values)
    else:
        X = df[[0]].values
    y = df[[1]].values
    return h_type(X, y)


def solve_btn(h, properties):
    if st.button('Solve', key='solve_btn'):
        st.write(h.weight)
        with st.spinner('waiting...'):
            loss_history, weights_history = properties.modification(h, properties.max_num_itter, properties.cost_function,
                                                                    regularization=properties.regularization, C=properties.reg_coef,
                                                                    alpha=properties.alpha, eps=properties.eps)
            st.success('Finished!')
        st.write(h.weight)
        cost_function_plot_2d(h, properties, loss_history, weights_history)
        

def gd_solution_page():
    st.title('Gradient Descent')
    properties = show_side_bar()
    task_type = select_task_type()

    if task_type == 'Individual':
        h = individual_task(properties.hypothesis, properties.scaler)
    elif task_type == 'Generate regression task':
        kwargs = params_for_generate_regression()
        h = generate_regression_task(properties.hypothesis, properties.scaler, **kwargs)
    elif task_type == 'Generate classification task':
        kwargs = params_for_generate_classification()
        h = generate_clasiffication_task(properties.hypothesis, properties.scaler, **kwargs)
    
    solve_btn(h, properties)
