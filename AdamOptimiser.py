import numpy as np

class AdamOptimiser(object):

    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
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
        self.update_params(layer_grad)
        return self.lr * self.m_hat / (np.sqrt(self.v_hat + self.epsilon)) # step 4
    
    def update_params(self, layer_grad):
        self.grad = layer_grad
        self.t += 1
        print(self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * self.grad # step 1
        self.v = self.beta2 * self.v + (1 - self.beta2) * (self.grad ** 2) # step 2
        self.m_hat = self.m / (1 - self.beta1 ** self.t) # step 3
        self.v_hat = self.v / (1 - self.beta2 ** self.t) # step 3
        return
