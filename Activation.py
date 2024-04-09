import numpy as np

class Activation(object):
    def tanh(self, x):
        return np.tanh(x)

    def tanh_deriv(self, a):
        # a = np.tanh(x)
        return 1.0 - a**2
    def logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def logistic_deriv(self, a):
        # a = logistic(x)
        return a * (1 - a)
        
    def ReLU(self, x):
        return x * (x > 0)
        
    def ReLU_deriv(self, a):
        return 1. * (a > 0)
    
    def softmax(self, x):
        exp_values = np.exp(x) # Computing element wise exponential value
        # exp_values_sum = np.sum(exp_values) # Computing sum of these values
        exp_values_sum = np.sum(exp_values, axis=-1, keepdims=True) # For Batch Computation - axis=-1 row-wise
        return exp_values / exp_values_sum # Returing the softmax output.
    
    def softmax_deriv(self, a): #from a website - needed??
        jacobian_m = np.diag(a)
        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = a[i] * (1-a[i])
                else: 
                    jacobian_m[i][j] = -a[i] * a[j]
        return jacobian_m

    def __init__(self, activation = "ReLU"):
        if activation == "logistic":
            self.fn = self.logistic
            self.fn_deriv = self.logistic_deriv
        elif activation == "tanh":
            self.fn = self.tanh
            self.fn_deriv = self.tanh_deriv
        elif activation == "ReLU":
            self.fn = self.ReLU
            self.fn_deriv = self.ReLU_deriv
        elif activation == "softmax":
            self.fn = self.softmax
            self.fn_deriv = self.softmax_deriv
