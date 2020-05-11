from gradient_descents import classic_grad_descent
from hypotheses import Linear
from cost_functions import MSE

from sklearn.preprocessing import StandardScaler, MinMaxScaler


MODIFICATIONS = {
    'Classic GD': classic_grad_descent,
    'SGD': classic_grad_descent,
    'SGD with Momentum': classic_grad_descent,
    'RMSProp': classic_grad_descent,
    'Adam': classic_grad_descent
}


HYPOTHESES = {
    'Linear': Linear,
    'Square': Linear
}


COST_FUNCTIONS = {
    'MSE': MSE,
    'MAE': MSE
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
