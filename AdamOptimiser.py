import numpy as np

class AdamOptimiser(object):

    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Creates a Adam's optimiser object for each hidden layer (since all different shapes eg (128, 64) vs (64, 32))

        Parameters:
        - lr (float): Learning rate.
        - beta1 (float): decay factor for momentum usually 0.9
        - beta2 (float): decay factor for infinity norm usually 0.999
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.v = 0
        self.grad = 0
        self.m = 0
        self.v_hat = 0
        self.m_hat = 0
        self.lr = lr
        self.epsilon = epsilon

    def update(self, layer_grad):
        """
        Calculates the adjustment needed for the weights or biases

        Parameters:
        - layer_grad (np.darray): Gradients of the weights/biases for the layer.

        Returns:
        - The adjustment needed to subtracted from the weight/bias
        """
        self.update_params(layer_grad)
        return self.lr * self.m_hat / (np.sqrt(self.v_hat + self.epsilon)) # step 4
    
    def update_params(self, layer_grad):
        """
        Updates the parameters of the Adam's Optimiser to be used in the next update of weights/biases

        Parameters:
        - layer_grad (np.darray): Gradients of the weights/biases for the layer.

        """
        self.grad = layer_grad
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * self.grad # step 1
        self.v = self.beta2 * self.v + (1 - self.beta2) * (self.grad ** 2) # step 2
        self.m_hat = self.m / (1 - self.beta1 ** self.t) # step 3
        self.v_hat = self.v / (1 - self.beta2 ** self.t) # step 3
        return
