import numpy as np
from Activation import *
from HiddenLayer import *
from MiniBatch import *
from AdamOptimiser import *

class MLP:
    def __init__(self, layers, activation=[None,'tanh','tanh', 'softmax'], use_batch_norm=False, weight_decay=1e-5):
        self.layers=[]
        self.params=[]
        self.activation=activation
        for i in range(len(layers)-1):
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1], use_batch_norm=use_batch_norm))

    def forward(self,input):
        for layer in self.layers: 
            output=layer.forward(input)
            input=output
        return output

    # Loss function. 
    def criterion_MSE(self, y, y_hat):
        """
        Mean Squared Error loss function.

        Parameters:
        - y (np.ndarray): True labels.
        - y_hat (np.ndarray): Predicted labels.

        Returns:
        - loss (float): The MSE loss.
        - delta (np.ndarray): Gradient of the loss with respect to y_hat.
        """
        error =  y - y_hat
        loss = np.mean(error**2)
        # activation_deriv=Activation(self.activation[-1]).fn_deriv
        # delta=-2*error*activation_deriv(y_hat)
        delta = -2 * error / y.shape[0]
        return loss, delta
    
    def softmax(self, values):
        exp_values = np.exp(values) # Computing element wise exponential value
        exp_values_sum = np.sum(exp_values) # Computing sum of these values
        return exp_values/exp_values_sum # Returing the softmax output.
        
    # def criterion_CrossEL(self, y, y_hat, epsilon=1e-9):
    #     loss = -np.sum(y * np.log(y_hat + epsilon)) / y.shape[0] # Adding a small value to avoid log(0)
    #     delta = y_hat - y
    #     return loss, delta
    
    def criterion_CrossEL(self, y, y_hat): #Nup, well defined functions - https://datascience.stackexchange.com/questions/20296/cross-entropy-loss-explanation
        loss = 0
        for j in range(len(y_hat)):
            loss += (-1 * y[j] * np.log(y_hat[j])) # y is the one-hot-vector
        delta = y_hat - y #Easy derivative calc - should I be using the other softmax deriv, dont think so
        return loss, delta

    # CrossEL Looks good. Only thing I was wondering is if it would be on the softmax of y_hat?
    # Tested with both but couldn't see a difference in the loss.
    # def criterion_CrossEL(self, y, y_hat):
    #     #y_hat = self.softmax(y_hat)
    #     loss = -np.sum(y * np.log(y_hat))
    #     delta = y_hat - y
    #     return loss, delta

    def backward(self, delta):
        """
        Performs backward pass through the entire network.

        Parameters:
        - delta (np.ndarray): Gradient of the loss with respect to the output of the last layer.
        """
        delta = self.layers[-1].backward(delta,output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta=layer.backward(delta)

    def update(self, lr, weight_decay):
        """
        Updates the weights and biases of all layers, including batch normalization parameters if used.

        Parameters:
        - lr (float): Learning rate.
        - weight_decay (float): Weight decay rate for neural network weights and bias
        """
        
        # do adams optimiser stuff
        for layer in self.layers:
            if layer.W_optimiser != None and layer.b_optimiser != None:
                layer.W = layer.W - layer.W_optimiser.update(layer.grad_W) #Could even generalise this by setting a default optimiser
                layer.b = layer.b - layer.b_optimiser.update(layer.grad_b)
            else:
                layer.W -= lr * layer.grad_W
                layer.W -= layer.W * weight_decay #currently doesnt weight decay in adam's
                layer.b -= lr * layer.grad_b
                layer.b -= layer.b * weight_decay
                if layer.use_batch_norm: # not sure if needed if using batch norm
                    layer.gamma -= lr * layer.grad_gamma
                    layer.gamma -= layer.gamma * weight_decay
                    layer.beta -= lr * layer.grad_beta
                    layer.beta -= layer.beta * weight_decay
            
            
    def fit(self,X,y, X_val, y_val,learning_rate=0.1, epochs=30, batch_size=32, weight_decay=1e-5, optimiser='Adam'): #this is normal when using 1. which is expected
        """
        Trains the MLP using the provided training data.

        Parameters:
        - X (np.ndarray): Training data features.
        - y (np.ndarray): Training data labels.
        - X_val (np.ndarray, optional): Validation data features.
        - y_val (np.ndarray, optional): Validation data labels.
        - learning_rate (float): Learning rate for the optimizer.
        - epochs (int): Number of epochs to train for.
        - batch_size (int): Size of the mini-batches for training.
        - weight_decay (float) : Weight decay rate for neural network weights and bias
        """
        if optimiser == 'Adam':
            for layer in self.layers:
                layer.set_optimiser(learning_rate)
        
        if X_val is not None and y_val is not None:
            validation_loss = np.zeros((epochs))
            
        MiniBatches = MiniBatch(X, y, batch_size)
        epoch_loss = np.zeros((epochs))

        for k in range(epochs): 
            MiniBatches.create_batches()
            loss_minibatch = np.zeros((epochs, MiniBatches.no_of_batches))
            
            for j in range(0, MiniBatches.no_of_batches): #for each batch - dont need this coz batches are randomised!
                X_minibatch = MiniBatches.batches_x[j]
                y_minibatch = MiniBatches.batches_y[j]
                loss = np.zeros((MiniBatches.batch_size))
                delta_minibatch = np.zeros((MiniBatches.batch_size, 10))
                
                for i in range(0, MiniBatches.batch_size): #for all in batch
                    y_hat = self.forward(X_minibatch[i]) # should be the same
                    loss[i], delta_minibatch[i] = self.criterion_CrossEL(y_minibatch[i], y_hat) 

                delta_avg = sum(delta_minibatch)/len(delta_minibatch)
                self.backward(delta_avg) #Think this is good, but dont know why loss increases
                self.update(learning_rate, weight_decay)
                loss_minibatch[k][j] = np.mean(loss)
                
            # Calculate the average loss for the epoch
            epoch_loss[k] = np.mean(loss_minibatch[k])
            
            # Calculate the validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                val_loss = np.zeros((X_val.shape[0]))
                for i in range(len(X_val)):
                    y_hat_val = self.forward(X_val[i]) #y_hat_val is not being used!!!
                    val_loss[i], _ = self.criterion_CrossEL(y_val[i], y_hat_val) 
                validation_loss[k] = np.mean(val_loss)
                
                print(f"Epoch {k+1}/{epochs}, Train loss: {epoch_loss[k]:.5f}, Val loss: {validation_loss[k]:.5f}")
            else:
                print(f"Epoch {k+1}/{epochs}, Train loss: {epoch_loss[k]:.5f}")

        return epoch_loss, validation_loss if validation_loss is not None else None

    def predict(self, x):
        x = np.array(x)
        output = np.zeros((x.shape[0],10))
        for i in range(len(x)):
            output[i] = self.forward(x[i])
        return np.array(output)