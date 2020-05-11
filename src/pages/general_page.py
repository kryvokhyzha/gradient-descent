import streamlit as st
import pickle as pk


def general_page():
    st.title('General page')
    st.text('''Gradient descent is a way to minimize an objective function J(θ) parameterized by a model’s \n
parameters θ ∈ R \n
d by updating the parameters in the opposite direction of the gradient of the  \n
objective function ∇θJ(θ) w.r.t. to the parameters. The learning rate η determines the size of the \n
steps we take to reach a (local) minimum. In other words, we follow the direction of the slope of the  \n
surface created by the objective function downhill until we reach a valley''')
