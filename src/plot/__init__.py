from plot.grid_compute import compute_j_grid, compute_j
from plot.plots_2d import cost_function_plot_2d, loss_plot_2d, data_plot_2d
from plot.plots_3d import data_plot_3d, cost_function_plot_3d


def plot_all(h, properties, weights_history, loss_history, y_pred_history):
    cost_function_plot_2d(h, properties, weights_history)
    cost_function_plot_3d(h, properties, weights_history)
    loss_plot_2d(loss_history)
    data_plot_2d(h, y_pred_history)
    data_plot_3d(h, y_pred_history)
