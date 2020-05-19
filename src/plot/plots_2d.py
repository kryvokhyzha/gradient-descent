import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st


def compute_j_grid(h, theta0_grid, theta1_grid, cost_function, C=1, regularization=None):
    if regularization is None:
        penalty = lambda x: (x * 0).sum()
    elif regularization == 'L1':
        penalty = lambda x: C*np.abs(x)[:, 1:].sum() / len(h.y)
    elif regularization == 'L2':
        penalty = lambda x: C * np.square(x)[:, 1:].sum() / (len(h.y)*2)

    grid = []
    for theta0 in theta0_grid:
        row = []
        for theta1 in theta1_grid:
            w = np.array([[theta0], [theta1]])
            y_pred = h.hypothesis(w=w)
            elem = cost_function.get_loss(y_pred, h.y) + penalty(w)
            row.append(elem)
        grid.append(row)
    return np.array(grid)


def cost_function_plot_2d(h, properties, weights_history, ax=None, fig=None):
    if h.X.shape[1] != 2 and len(weights_history) > 2:
        return

    if ax is None or fig is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6.15))

    x = h.X[:, 1]
    y = list(map(lambda x: x[0], h.y))

    theta = [np.array([i[0][0], i[1][0]]) for i in weights_history]

    theta0 = [i[0] for i in weights_history]
    theta1 = [i[1] for i in weights_history]

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

#    data = go.Contour(
#        z=J_grid,
#        contours=dict(
#            coloring='lines',
#            showlabels=True, # show labels on contours
#            labelfont=dict( # label font properties
#                size=12,
#                color='black',
#            ))
#    )
    
#    fig = go.Figure(data=data)
#    st.plotly_chart(fig)


def loss_plot_2d(loss_history):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(1, len(loss_history)+1)), y=loss_history,
                    opacity=1,
                    line=dict(color='firebrick', width=1),
                    mode='lines+markers'))

    fig.update_layout(title='Cost function value for each iteration', autosize=False,
                      width=800,
                      height=600,
                      xaxis_title='num_iter',
                      yaxis_title='J(w)')
    st.plotly_chart(fig)


def data_plot_2d(h, y_pred_history):
    if h.X.shape[1] != 2:
        return

    x = h.X[:, 1]
    y = list(map(lambda x: x[0], h.y))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=y,
                    mode='markers',
                    name='data'))

    tr = 0.1
    for i in range(len(y_pred_history)):
        if i % 25 == 0:
            fig.add_trace(go.Scatter(x=x, y=list(map(lambda x: x[0], y_pred_history[i])),
                    opacity=tr,
                    line=dict(color='firebrick', width=2),
                    mode='lines',
                    showlegend=False,
                    name=f'{1 if i == 0 else i} iteration'))

            tr += 14 / len(y_pred_history)

    fig.add_trace(go.Scatter(x=x, y=list(map(lambda x: x[0], y_pred_history[i])),
                    opacity=1,
                    line=dict(color='firebrick', width=2),
                    mode='lines',
                    showlegend=True,
                    name=f'approximating curve'))

    fig.update_layout(title='Data scatter plot and Approxomating curve', autosize=False,
                      width=900,
                      height=600,
                      xaxis_title='X',
                      yaxis_title='y')

    st.plotly_chart(fig)
