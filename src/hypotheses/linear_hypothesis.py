import numpy as np
from hypotheses import Abstract_Hypothesis 


class Linear(Abstract_Hypothesis):

    def __init__(self, X, y):
        self.X = np.hstack((np.ones((len(X), 1)), X))
        self.y = y 
        self.weight = np.random.normal(size=(self.X.shape[1], 1))
           
    def hypothesis(self):
        return np.dot(self.X, self.weight)

    def hypothesis_grad(self):
        return self.X
