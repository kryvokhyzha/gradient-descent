import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from plot import compute_j_grid, compute_j


def cost_function_plot_3d(h, properties, weights_history, loss_history):
    if h.weight.shape[0] != 2 and len(weights_history) > 2:
        return

    y = list(map(lambda x: x[0], h.y))

    theta = [np.array([i[0][0], i[1][0]]) for i in weights_history]

    theta0 = np.array([i[0][0] for i in weights_history])
    theta1 = np.array([i[1][0] for i in weights_history])

    thetha0_min = np.min(theta0) - 3*np.std(theta0)
    thetha0_max = np.max(theta0) + 3*np.std(theta0)

    thetha1_min = np.min(theta1) - 3*np.std(theta1)
    thetha1_max = np.max(theta1) + 3*np.std(theta1)

    theta0_grid = np.linspace(thetha0_min, thetha0_max, 101)
    theta1_grid = np.linspace(thetha1_min, thetha1_max, 101)

    J_grid = compute_j_grid(h, theta0_grid, theta1_grid, properties.cost_function, C=properties.reg_coef, regularization=properties.regularization)

    X, Y = np.meshgrid(theta0_grid, theta1_grid)

    fig = go.Figure()

    fig.add_trace(
        go.Surface(x=X, y=Y, z=J_grid, colorscale='RdBu', showscale=False))

    fig.add_trace(go.Scatter3d(x=theta0[1:], y=theta1[1:], z=loss_history[1:], mode='lines+markers',
                    marker=dict(
                        size=3,
                        color='green',
                        #color=y,                # set color to an array/list of desired values
                       # colorscale='greens',   # choose a colorscale
                        opacity=0.8
                    )))
    
    fig.update_layout(title='Cost function surface plot', autosize=False,
                      width=900,
                      height=600)

    st.plotly_chart(fig)


def data_plot_3d(h, y_pred_history):
    if h.X_raw.shape[1] != 3:
        return

    x = h.X_raw[:, 1:]
    y = list(map(lambda x: x[0], h.y))

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=x[:, 0], y=x[:, 1], z=y, mode='markers',
                    marker=dict(
                        size=3,
                        color=y,                # set color to an array/list of desired values
                        colorscale='Viridis',   # choose a colorscale
                        opacity=0.8
                    )))

    xGrid, yGrid = np.meshgrid(x[:, 0], x[:, 1])
    l = len(xGrid)
    Z = np.array([h.hypothesis(X=np.vstack(([1]*l, xGrid[i], yGrid[i])).T).reshape(1, l)[0] for i in range(l)])

    fig.add_trace(
        go.Surface(x=x[:, 0], y=x[:, 1], z=Z, colorscale='RdBu', showscale=False))

    fig.update_layout(title='Data 3D scatter plot and Approxomating curve', autosize=False,
                      width=900,
                      height=600)

    st.plotly_chart(fig)
