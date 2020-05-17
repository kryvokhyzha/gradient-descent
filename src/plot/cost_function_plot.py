import numpy as np
import matplotlib.pyplot as plt


def cost_function_plot():
    theta0_grid = np.linspace(-1,4,101)
    theta1_grid = np.linspace(-5,5,101)
    J_grid = cost_func(theta0_grid[np.newaxis,:,np.newaxis],
                    theta1_grid[:,np.newaxis,np.newaxis])