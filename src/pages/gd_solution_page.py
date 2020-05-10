import streamlit as st
import pandas as pd
import os

from gradient_descents import classic_grad_descent
from hypotheses import Linear
from cost_functions import MSE


def gd_solution_page():
    st.title('Gradient Descent')
    if st.button('Solve', key='solve_btn'):
        df = pd.read_csv('data/restaurant_revenue.txt', header=None, sep=',')
        st.write(df[[0]])
        st.write(df[[1]])
        linear = Linear(df[[0]], df[[1]])
        st.write(linear.weight)
        classic_grad_descent(linear, 1000, MSE)
        st.write(linear.weight)
