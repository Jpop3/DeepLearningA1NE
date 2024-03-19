import numpy as np
from Activation import *
from HiddenLayer import *
from MiniBatch import *

class MLP:
    def __init__(self, layers, activation=[None,'tanh','tanh']):
        self.layers=[]
        self.params=[]
        self.activation=activation
        for i in range(len(layers)-1):
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1]))

    def forward(self,input):
        for layer in self.layers: 
            output=layer.forward(input)
            input=output
        return output

    #Loss function
    def criterion_MSE(self, y, y_hat):
        activation_deriv=Activation(self.activation[-1]).fn_deriv
        error = y-y_hat
        loss=error**2
        delta=-2*error*activation_deriv(y_hat)
        return loss,delta
    
    def softmax(self, values):
        exp_values = np.exp(values) # Computing element wise exponential value
        exp_values_sum = np.sum(exp_values) # Computing sum of these values
        return exp_values/exp_values_sum # Returing the softmax output.
    
    def criterion_CrossEL(self, class_label, y_hat): #Nup, well defined functions
        loss = 0
        for j in range(len(y_hat)):
            loss += (-1 * class_label[j] * np.log(y_hat[j])) #class_label is the one-hot-vector
        delta = y_hat - class_label #Easy derivative calc - should I be using the other softmax deriv, dont think so
        return loss, delta

    def backward(self, delta):
        delta = self.layers[-1].backward(delta,output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta=layer.backward(delta)

    def update(self,lr):
        for layer in self.layers:
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b

    #Training
    # def fit(self, X, y, learning_rate=0.1, epochs=100):
    #     X=np.array(X)
    #     y=np.array(y)
    #     to_return = np.zeros(epochs)

    #     for k in range(epochs): 
    #         loss=np.zeros(X.shape[0])
    #         for it in range(X.shape[0]): #Mini-batch training should be done here
    #             i = np.random.randint(X.shape[0]) #taking a random input from the training data

    #             y_hat = self.forward(X[i]) #use input i!

    #             loss[it], delta = self.criterion_CrossEL(y[i], y_hat) #input class labels, and prediced class labels

    #             self.backward(delta)
    #             y
    #             self.update(learning_rate)
    #         to_return[k] = np.mean(loss)
    #         print(to_return[k])
    #     return to_return
            
    def fit(self,X,y,learning_rate=0.1, epochs=30, batch_size=1): #this is normal when using 1. which is expected
        X=np.array(X)
        y=np.array(y)
        MiniBatches = MiniBatch(X, y, batch_size)
        epoch_loss = np.zeros((epochs))

        #print(MiniBatches.no_of_batches)
        #print(MiniBatches.batch_size)

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
                    loss[i], delta_minibatch[i] = self.criterion_CrossEL(y_minibatch[i], y_hat) #check loss function - ind loss for training example
                #     print("delta_minibatch[i] = {}".format(delta_minibatch[i]))
                #     print("loss = {}".format(loss[i]))
                #     print("y = {}".format(y_minibatch[i]))
                #     print("y_hat = {}".format(y_hat)) #exploding weights problem??
                # print(sum(delta_minibatch)/len(delta_minibatch))

                self.backward(sum(delta_minibatch)/len(delta_minibatch)) #Think this is good, but dont know why loss increases

                self.update(learning_rate)
                loss_minibatch[k][j] = np.mean(loss)
            epoch_loss[k] = np.mean(loss_minibatch[k])
            print(epoch_loss[k])
        return epoch_loss

    def predict(self, x):
        x = np.array(x)
        output = np.zeros((x.shape[0],10))
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i,:])
        return output