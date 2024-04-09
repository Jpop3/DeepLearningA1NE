import numpy as np
from Optimiser import *

class Momentum(Optimiser):

    def __init__(self, lr, gamma=0.9):
        """
        Is a child class of Optimiser
        Creates a Adam's optimiser object for each hidden layer (since all different shapes eg (128, 64) vs (64, 32))

        Parameters:
        - lr (float): Learning rate.
        - gamma (float): momentum constant usually 0.9
        """
        super().__init__(lr)
        self.gamma = gamma
        self.v = 0
        self.grad = 0

    def update(self, layer_grad):
        """
        Calculates the adjustment needed for the weights or biases

        Parameters:
        - layer_grad (np.darray): Gradients of the weights/biases for the layer.

        Returns:
        - The adjustment needed to subtracted from the weight/bias
        """
        self.update_params(layer_grad)
        return self.v
    
    def update_params(self, layer_grad):
        """
        Updates the parameters for Momentum to be used in the next update of weights/biases

        Parameters:
        - layer_grad (np.darray): Gradients of the weights/biases for the layer.
        """
        self.grad = layer_grad
        self.v = self.gamma * self.v + (self.lr * self.grad)
        return