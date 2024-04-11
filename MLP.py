import numpy as np
from Activation import *
from HiddenLayer import *
from MiniBatch import *

class MLP:
    def __init__(self, layers, activation=[None,'tanh','tanh', 'softmax'], use_batch_norm=0, weight_decay=1e-5, dropout_rate=[0.0, 0.0, 0.0, 0.0]):
        self.layers=[]
        self.params=[]
        self.activation=activation
        self.activation[0] = None #Fixing the "None" issues, just a hot fix
        for i in range(len(layers)-1):
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1], use_batch_norm=use_batch_norm, dropout_rate=dropout_rate[i]))
    
    # Static class to one hot function
    def class_to_one_hot(class_label, num_classes):
        one_hot = np.zeros(num_classes)
        one_hot[class_label] = 1.
        return one_hot

    def forward(self, input, train=False):
        for layer in self.layers: 
            output=layer.forward(input, train=train)
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
        # Shift values by subtracting the max value to prevent overflow
        values_shifted = values - np.max(values, axis=1, keepdims=True)
        exp_values = np.exp(values_shifted)
        exp_values_sum = np.sum(exp_values, axis=1, keepdims=True)
        # exp_values = np.exp(values) # Computing element wise exponential value
        # exp_values_sum = np.sum(exp_values) # Computing sum of these values
        return exp_values/exp_values_sum # Returing the softmax output.
        
    def criterion_CrossEL(self, y, y_hat, epsilon=1e-9):
        loss = -np.sum(y * np.log(y_hat + epsilon)) # Adding a small value to avoid log(0)
        delta = y_hat - y
        return loss, delta
    
    def criterion_CrossEL_Batch(self, y, y_hat, epsilon=1e-9):
        # Compute the cross-entropy loss for each instance in the batch
        loss = -np.sum(y * np.log(y_hat + epsilon), axis=-1) # axis=-1 row-wise
        # Calculate the average loss over the batch
        average_loss = np.mean(loss)
        # Compute the gradient of the loss w.r.t. the predictions (y_hat)
        delta = y_hat - y
        return average_loss, delta

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
        
        for layer in self.layers:
            layer.W = layer.W - layer.W_optimiser.update(layer.grad_W)
            layer.b = layer.b - layer.b_optimiser.update(layer.grad_b)
            if weight_decay != 0:
                layer.W -= layer.W * weight_decay
                layer.b -= layer.b * weight_decay
            if layer.use_batch_norm: # not sure if needed if using batch norm with certain optimisers
                layer.gamma -= lr * layer.grad_gamma
                layer.gamma -= layer.gamma * weight_decay
                layer.beta -= lr * layer.grad_beta
                layer.beta -= layer.beta * weight_decay
            
            
    def fit(self,X,y, X_val, y_val,learning_rate=0.1, epochs=30, batch_size=32, weight_decay=1e-5, optimiser='Adam', early_stopping=(10, 0.001), verbose=True): #this is normal when using 1. which is expected
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
        for layer in self.layers:
            layer.set_optimiser(optimiser, learning_rate)
        
        # Initialize the validation loss array if validation data is provided
        if X_val is not None and y_val is not None:
            validation_loss = np.zeros((epochs))
            
        # Early stopping object
        if early_stopping is not None and X_val is not None and y_val is not None:
            stop_early = self.EarlyStopping(*early_stopping)
            
        MiniBatches = MiniBatch(X, y, batch_size)
        epoch_loss = np.zeros((epochs))

        for k in range(epochs): 
            MiniBatches.create_batches()
            loss_minibatch = np.zeros((epochs, MiniBatches.no_of_batches))
            
            for j in range(0, MiniBatches.no_of_batches): #for each batch - dont need this coz batches are randomised!
                X_minibatch = MiniBatches.batches_x[j]
                y_minibatch = MiniBatches.batches_y[j]
                
                ######## Update to use vectorized implementation ########
                # loss = np.zeros((MiniBatches.batch_size))
                # delta_minibatch = np.zeros((MiniBatches.batch_size, 10))
                # for i in range(0, MiniBatches.batch_size): # for all in batch
                #    y_hat = self.forward(X_minibatch[i], train=True) # apply forward pass (training mode for dropout)
                #    loss[i], delta_minibatch[i] = self.criterion_CrossEL(y_minibatch[i], y_hat)
                # delta_avg = sum(delta_minibatch)/len(delta_minibatch)
                # self.backward(delta_avg) #Think this is good, but dont know why loss increases

                # Perform foward pass on the minibatch using vectorized implementation
                y_hat = self.forward(X_minibatch, train=True)
                loss, delta_minibatch = self.criterion_CrossEL_Batch(y_minibatch, y_hat)
                self.backward(delta_minibatch)
                
                self.update(learning_rate, weight_decay)
                loss_minibatch[k][j] = np.mean(loss)
                
            # Calculate the average loss for the epoch
            epoch_loss[k] = np.mean(loss_minibatch[k])
            
            # Calculate the validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self.criterion_CrossEL_Batch(y_val, val_pred)[0]
                # val_loss = np.mean([self.criterion_CrossEL(y_val[i], val_pred[i])[0] for i in range(len(X_val))])
                validation_loss[k] = val_loss
                
                # Check for early stopping
                if early_stopping is not None:
                    stop_early(val_loss, k)
                    if stop_early.should_stop:
                        print(f"\tearly stopping at ep.{k+1}\t train loss: {epoch_loss[k]:.5f}\t val loss: {val_loss:.5f}\t best val loss: {stop_early.best_score:.5f}")
                        break
                
                if verbose:
                    print(f"epoch: {k+1}/{epochs}\t train loss: {epoch_loss[k]:.5f}\t val loss: {val_loss:.5f}")
            else:
                if verbose:
                    print(f"epoch: {k+1}/{epochs}\t train loss: {epoch_loss[k]:.5f}")
        
        if validation_loss is not None:
            if early_stopping is not None and stop_early.should_stop:
                return epoch_loss[:k+1], validation_loss[:k+1], k+1
            else:
                return epoch_loss, validation_loss, None
        else:
            return epoch_loss, None, None

    def predict(self, x):
        x = np.array(x)
        # output = np.zeros((x.shape[0],10))
        # for i in range(len(x)):
        #     output[i] = self.forward(x[i], train=False) # dropout off during inference
        output = self.forward(x, train=False)
        return np.array(output)
    
    
    # Inner class for early stopping
    class EarlyStopping:
        def __init__(self, patience=10, min_delta=0.001):
            """
            Initializes the EarlyStopping instance.
            Parameters:
                patience (int): The number of epochs to wait for improvement before stopping the training.
                min_delta (float): The minimum change in the monitored metric to qualify as an improvement.
            """
            self.patience = patience
            self.min_delta = min_delta
            self.best_score = None
            self.epochs_without_improvement = 0
            self.should_stop = False

        def __call__(self, current_val_loss, epoch):
            """
            Call method to update the early stopping logic.
            Parameters:
                current_val_loss (float): The current epoch's validation loss.
            """
            if self.best_score is None or current_val_loss < self.best_score - self.min_delta:
                self.best_score = current_val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    self.should_stop = True
            # Early stop for high validation loss
            # if epoch > 20 and current_val_loss >= 2 and self.best_score >= 2:
            #     self.should_stop = True