import numpy as np
from hypotheses import Abstract_Hypothesis 


class Linear(Abstract_Hypothesis):

    def __init__(self, X, y):
        self.X = np.hstack((np.ones((len(X), 1)), X))
        self.y = y 
        self.weight = np.random.normal(size=(self.X.shape[1], 1))
           
    def hypothesis(self, X=None, w=None):
        if w is None:
            w = self.weight

        if X is None:
            X = self.X

        return np.dot(X, w)

    def hypothesis_grad(self):
        return self.X
