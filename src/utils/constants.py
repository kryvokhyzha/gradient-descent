from gradient_descents import classic_grad_descent, stochastic_grad_descent, momentum_grad_descent, rmsprop_grad_descent, adam_grad_descent
from hypotheses import Linear, Sigmoid, Polynomial
from cost_functions import MSE, MAE, BCE

from sklearn.preprocessing import StandardScaler, MinMaxScaler


MODIFICATIONS = {
    'Classic GD': classic_grad_descent,
    'SGD': stochastic_grad_descent,
    'SGD with Momentum': momentum_grad_descent,
    'RMSProp': rmsprop_grad_descent,
    'Adam': adam_grad_descent
}


HYPOTHESES = {
    'Linear': Linear,
    'Polynomial': Polynomial,
    'Sigmoid': Sigmoid
}


COST_FUNCTIONS = {
    'MSE': MSE,
    'MAE': MAE,
    'BCE': BCE
}

REGULARIZATION = {
    'None': None,
    'L1': 'L1',
    'L2': 'L2'
}

SCALE = {
    'None': None,
    'StandartScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler()
}
