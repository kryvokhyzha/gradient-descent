import streamlit as st
import numpy as np
from helper import markdown_to_string
from plot import loss_plot_2d, execution_time_plot_2d


def comparison_page():
    st.markdown(markdown_to_string('data/markdown/comparison_page.md'))

    with open('data/loss_history.npy', 'rb') as f:
        loss_h = np.load(f)

    loss_plot_2d(loss_h)

    with open('data/execution_time.npy', 'rb') as f:
        time_h = np.load(f)

    execution_time_plot_2d(time_h)
