import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from plot import compute_j_grid


def cost_function_plot_2d(h, properties, weights_history):
    if h.weight.shape[0] != 2 and len(weights_history) > 2:
        return

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6.15))

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
    contours = ax.contour(X, Y, J_grid, 30, lable='level lines')

    ax.clabel(contours)
    for j in range(2,len(theta)):
        ax.annotate('', xy=theta[j], xytext=theta[j-1],
                    arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                    va='center', ha='center', label='each GD step')
    
    ax.set_xlabel(r'$w_0$')
    ax.set_ylabel(r'$w_1$')
    ax.set_title('Cost function contour plot')
    ax.legend()

    plt.show()
    st.pyplot()


def loss_plot_2d(loss_history):
    fig = go.Figure()

    if len(loss_history.shape) == 1:
        title = 'Cost function value for each iteration'
        fig.add_trace(go.Scatter(x=list(range(1, len(loss_history)+1)), y=loss_history,
                        opacity=1,
                        line=dict(color='firebrick', width=1),
                        mode='lines+markers'))
    else:
        title = ''
        colors = ['green', 'red','yellow', 'cyan', 'magenta']
        algorithm_name = ['Classic GD', 'SGD', 'Momentum', 'Rmsprop', 'Adam']
        i=0
        for loss in loss_history:
            fig.add_trace(go.Scatter(x=list(range(1, len(loss)+1)), y=loss,
                            opacity=1,
                            line=dict(color=colors[i], width=3),
                            mode='lines', name = algorithm_name[i]))
            i+=1


    fig.update_layout(title=title, autosize=False,
                      width=800,
                      height=600,
                      xaxis_title='num_iter',
                      yaxis_title='J(w)')
    st.plotly_chart(fig)


def execution_time_plot_2d(time_history):
    fig = go.Figure()

    colors = ['green', 'red', 'yellow']
    algorithm_name = ['Classic GD', 'SGD', 'Momentum']

    for i, time in enumerate(time_history):
        fig.add_trace(go.Scatter(x=list(range(1, len(time)+1)), y=time,
                            line=dict(color=colors[i], width=3),
                            mode='lines', name=algorithm_name[i]))
    
    fig.update_layout(title='Execution time for each iteration', autosize=False,
                      width=800,
                      height=600,
                      xaxis_title='num_iter',
                      yaxis_title='execution_time')
    
    st.plotly_chart(fig)


def data_plot_2d(h, y_pred_history):
    if h.X_raw.shape[1] != 2:
        return

    x = h.X_raw[:, 1]
    y = list(map(lambda x: x[0], h.y))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=y,
                    mode='markers',
                    name='data'))

    tr = 0.1
    for i in range(len(y_pred_history)):
        if i % 25 == 0:
            y_pred_history_temp = list(map(lambda x: x[0], y_pred_history[i]))
            XY = list(zip(x, y_pred_history_temp))
            XY.sort(key=lambda x: x[0])

            fig.add_trace(go.Scatter(x=[i[0] for i in XY], y=[i[1] for i in XY],
                    opacity=tr,
                    line=dict(color='firebrick', width=2),
                    mode='lines',
                    showlegend=False,
                    name=f'{1 if i == 0 else i} iteration'))

            tr += 14 / len(y_pred_history)
    
    y_pred_history_temp = list(map(lambda x: x[0], y_pred_history[i]))
    XY = list(zip(x, y_pred_history_temp))
    XY.sort(key=lambda x: x[0])

    fig.add_trace(go.Scatter(x=[i[0] for i in XY], y=[i[1] for i in XY],
                    opacity=1,
                    line=dict(color='green', width=3),
                    mode='lines',
                    showlegend=True,
                    name=f'approximating curve'))

    fig.update_layout(title='Data scatter plot and Approxomating curve', autosize=False,
                      width=900,
                      height=600,
                      yaxis=dict(range=[-5,25]),
                      xaxis_title='X',
                      yaxis_title='y')

    st.plotly_chart(fig)
