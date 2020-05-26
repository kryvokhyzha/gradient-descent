from abc import ABC, abstractmethod
import numpy as np 


class Abstract_Hypothesis(ABC):

    def __init__(self, X, y, w=None, degree=1):
        super().__init__()
           
    @abstractmethod
    def hypothesis(self):
        pass
    
    @abstractmethod
    def hypothesis_grad(self):
        pass
