import numpy as np
from Activation import *
from Optimiser import *

class HiddenLayer(object):
    def __init__(self, n_in, n_out, activation_last_layer=None, \
        activation=None, W=None, b=None, use_batch_norm=False, \
        dropout_rate=0.0): #inputs for the hidden layer
        
        self.input=None
        self.activation=Activation(activation).fn
        self.activation_deriv=None 
        if activation_last_layer:
            self.activation_deriv=Activation(activation_last_layer).fn_deriv #set the derivative
            
        # Dropout rate and mask
        self.dropout_rate = dropout_rate
        self.mask = None

        # Batch normalization stuff
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.gamma = np.ones((n_out,))
            self.beta = np.zeros((n_out,))
            self.bn_cache = {} # A cache to store variables for backward pass

        # initialize weights and biases
        self.W = np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
        )
        self.b = np.zeros(n_out,)
        
        # Calculate the gradients of the weights, biases and batch norm parameters
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)
        
        if self.use_batch_norm:
            self.grad_gamma = np.zeros(self.gamma.shape)
            self.grad_beta = np.zeros(self.beta.shape)

        self.W_optimiser = None
        self.b_optimiser = None

    def set_optimiser(self, optimiser, learning_rate):
        if optimiser == 'Adam':
            self.W_optimiser = AdamOptimiser(learning_rate)
            self.b_optimiser = AdamOptimiser(learning_rate)
        elif optimiser == 'Momentum':
            self.W_optimiser = Momentum(learning_rate)
            self.b_optimiser = Momentum(learning_rate)
        else:
            self.W_optimiser = Optimiser(learning_rate)
            self.b_optimiser = Optimiser(learning_rate)

    def forward(self, input, train=False):
        self.input=input
        lin_output = np.dot(input, self.W) + self.b #Basic dot product
        
        # Batch normalization step
        if self.use_batch_norm:
            lin_output, self.bn_cache = self.batch_norm_forward(lin_output, self.gamma, self.beta)
        
        # Dropout step applied if training and dropout rate > 0
        if train and self.dropout_rate > 0:
            self.mask = (np.random.rand(*lin_output.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            lin_output *= self.mask
        
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output) #Activation function
        )
        return self.output
        

    def backward(self, delta, output_layer=False): #Backwards propagation
        '''
        Performs the backward pass for the hidden layer, computing gradients of the weights, biases and batch norm params (if used).
        
        Paramaters:
            delta (ndarray): Gradient of the loss with respect to the output of the layer
            output_layer (bool): Whether the layer is the output layer or not
        '''
        # Apply dropout mask
        if self.mask is not None:
            delta *= self.mask
        
        # Backpropagate the gradient through the activation function
        if self.activation_deriv and not output_layer:
            delta = delta * self.activation_deriv(self.output) # Derivative of the activation function
        
        # Backpropagate the gradient through the batch normalization layer
        if self.use_batch_norm:
            delta, self.grad_gamma, self.grad_beta = self.batch_norm_backward(delta, self.bn_cache)
        
        # print(delta.shape, self.W.shape, self.input.shape, self.input.T.shape)
        # Calculate the gradients of the weights and biases
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = np.sum(delta, axis=0) # Sum the gradients along the batch axis
        
        delta = delta.dot(self.W.T) * (self.activation_deriv(self.input) if self.activation_deriv else 1)
        return delta
    
    # def backward(self, delta, output_layer=False): #Backwards propagation
    #     self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
    #     self.grad_b = delta
    #     if self.activation_deriv:
    #         delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
    #     return delta
    
    def batch_norm_forward(self, X, gamma, beta, eps=1e-5):
        '''
        Forward pass for batch normalization.
        
        Parameters:
            X (ndarray): Input data for the layer before batch norm
            gamma (ndarray): Scale parameter
            beta (ndarray): Shift parameter
            eps (float): Small constant to avoid division by zero
            
        Returns:
            out (ndarray): Output of the batch normalization layer
            cache (tuple): Intermediate variables for backpropagation
        '''
        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)
        
        # Normalize the batch data
        X_norm = (X - mu) / np.sqrt(var + eps)
        
        # Scale and shift the normalized data using gamma and beta
        out = gamma * X_norm + beta
        
        # Store intermediate variables for backpropagation
        cache = (X, X_norm, mu, var, gamma, beta, eps)
        
        return out, cache
    
    def batch_norm_backward(self, grad_out, cache):
        '''
        Performs the backward pass of batch normalization
        
        Parameters:
            grad_out (ndarray): Gradient of the loss with respect to the output of the batch normalization layer
            cache (tuple): Intermediate variables from the forward pass
            
        Returns:
            grad_X (ndarray): Gradient of the loss with respect to the input of the batch normalization layer
            grad_gamma (ndarray): Gradient of the loss with respect to the scale parameter
            grad_beta (ndarray): Gradient of the loss with respect to the shift parameter
        '''
        
        X, X_norm, mu, var, gamma, beta, eps = cache
        N = X.shape[0]
        
        # Compute the gradients of gamma and beta
        grad_gamma = np.sum(grad_out * X_norm, axis=0)
        grad_beta = np.sum(grad_out, axis=0)
        
        # Compute the gradient of the loss with respect to the normalized input
        grad_X_norm = grad_out * gamma
        
        # Compute the gradients of mu and var.
        grad_var = np.sum(grad_X_norm * (X - mu), axis=0) * -0.5 * (var + eps)**(-1.5) # Derivative of the variance.
        # last term is the derivative of the square root of the variance. ???
        grad_mu = np.sum((grad_X_norm * -1) / np.sqrt(var + eps), axis=0) + grad_var * (np.sum(-2 * (X - mu), axis=0) / N) # Derivative of the mean.
        
        # Compute the gradient of the loss with respect to the input
        grad_X = (grad_X_norm / np.sqrt(var + eps)) + (grad_var * 2 * (X - mu) / N) + (grad_mu / N)
        return grad_X, grad_gamma, grad_beta