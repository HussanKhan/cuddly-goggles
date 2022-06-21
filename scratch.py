import numpy as np
import math

from torch import sigmoid

np.random.seed(1)

class Layer():
    
    
    def __init__(self, name, r, c, lr):
        self.lr = lr
        self.name = name
        self.matrix = np.random.rand(r,c)
        self.bias = np.random.rand(1, c)
        
        self.lastOutput = None
        self.lastInput = None
        
        
    def forward(self, x, final=False):
        
        self.lastInput = x
        
        if not final:
            self.lastOutput = self.sigmoid(x.dot(self.matrix))
        else:
            self.lastOutput = x.dot(self.matrix)
            
        return self.lastOutput
    
    # applies derv
    def backward(self, errorFromNext):
        print(self.name)
        print(self.matrix.shape)
        
        # apply changes to self
        delta = errorFromNext * np.array([self.lastInput]).T
        self.matrix -= (delta * self.lr)
        
        # create error for previous layer
        hidden_error = self.matrix.dot(errorFromNext)
        hidden_error_term = hidden_error * self.sigmoid_derv(self.lastInput)
        
        return hidden_error_term
            
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

        
        
        
x = np.random.randn(56,)

lr = 0.04
modelfc1 = Layer("Input", 56, 20, lr) # -> 5,5
modelfc3 = Layer("Hidden 1", 20,20, lr) # -> 5,1
modelfc4 = Layer("Output", 20,1, lr) # -> 5,1

x = modelfc1.forward(x)
x = modelfc3.forward(x)
x = modelfc4.forward(x, final=True)

y = np.random.randn(1,)

error = y - modelfc4.lastOutput

e = modelfc4.backward(error)
e = modelfc3.backward(e)
e = modelfc1.backward(e)