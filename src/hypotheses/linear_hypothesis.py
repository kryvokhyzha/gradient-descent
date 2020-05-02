import numpy as np
from hypotheses import Abstract_Hypothesis 


class Linear(Abstract_Hypothesis):

    def __init__(self, X, y):
        self.X = np.hstack((np.ones((len(X), 1)), X))
        self.y = y 
        self.weigth = np.random.normal(size=(1, X.shape[1]))
           
    def hypothesis(self):
        return self.weigth @ self.X

    def hypothesis_grad(self):
        return self.X
