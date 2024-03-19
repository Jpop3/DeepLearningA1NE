import numpy as np
from Activation import *

class HiddenLayer(object):
    def __init__(self, n_in, n_out,
                 activation_last_layer=None, activation=None, W=None, b=None): #inputs for the hidden layer
        
        self.input=None
        self.activation=Activation(activation).fn

        self.activation_deriv=None 
        if activation_last_layer:
            self.activation_deriv=Activation(activation_last_layer).fn_deriv #set the derivative

        self.W = np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
        )
        
        self.b = np.zeros(n_out,)
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, input):
        lin_output = np.dot(input, self.W) + self.b #Basic dot product
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output) #Activation function
        )
        self.input=input
        return self.output

    def backward(self, delta, output_layer=False): #Backwards propagation
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = delta
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        return delta