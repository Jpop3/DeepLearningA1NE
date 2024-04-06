import numpy as np
import pandas as pd
from MLP import *

def class_to_one_hot(class_label, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[class_label] = 1.
    return one_hot

def train(nn, input, labels, lr=0.001, epochs=20, batch_size=2):
    """
    Trains with a validation set

    Parameters:
    - nn (MLP): Neural Network Model.
    - input (np.darray (50000, 128)): input data
    - labels (np.darray (50000, 10)): output class labels
    """
    #Make one hot labels
    one_hot_labels = np.zeros((50000, 10))
    for index, label in enumerate(labels):
        one_hot_labels[index] = class_to_one_hot(label, 10)

    ## Create training and validation sets from randomly shuffled data
    indices = np.random.permutation(input.shape[0])
    training_idx, val_idx = indices[:40000], indices[40000:]
    input_training, input_val = input[training_idx,:], input[val_idx,:]
    labels_training, labels_val = one_hot_labels[training_idx,:], one_hot_labels[val_idx,:]

    input_training = np.array(input_training)
    labels_training = np.array(labels_training)
    input_val = np.array(input_val)
    labels_val = np.array(labels_val)

    CEL = nn.fit(input_training, labels_training, input_val, labels_val, learning_rate=lr, epochs=epochs, batch_size=batch_size) 

    get_training_accuracy(nn, input_training, labels_training)
    return

def get_training_accuracy(nn, input_training, labels_training):
    output = nn.predict(input_training)
    correct_count = 0
    for index, array in enumerate(output):
        if np.argmax(array) == np.argmax(labels_training[index]): #the largest val in the vector indicates what class its predicting.
            correct_count += 1
    print("Training accuracy = {}".format(correct_count/40000))

def get_test_accuracy(nn, input_test, labels_test):
    test_one_hot_labels = np.zeros((10000, 10)) 
    for index, label in enumerate(labels_test):
        test_one_hot_labels[index] = class_to_one_hot(label, 10) #Make one hot labels
    output = nn.predict(input_test)
    correct_count = 0
    for index, array in enumerate(output):
        if np.argmax(array) == np.argmax(test_one_hot_labels[index]):
            correct_count += 1
    print("Test accuracy = {}".format(correct_count/10000))

input = np.load('Assignment1-Dataset/train_data.npy')
labels = np.load('Assignment1-Dataset/train_label.npy')

# Set numpy seed for reproducibility
np.random.seed(1234)
nn = MLP([128, 64, 32, 10], [None, 'ReLU', 'ReLU', 'softmax'], use_batch_norm=True) #can adjust hidden layers and activation functions
train(nn, input, labels)

input_test = np.load('Assignment1-Dataset/test_data.npy') #Load datasets
labels_test = np.load('Assignment1-Dataset/test_label.npy') #Load datasets

get_test_accuracy(nn, input_test, labels_test)


#Loss jumped back up?? after hitting 1.30 to 1.61 and more - this was because wasnt shuffling minibatches
#Internet said make the learning rate smaller. but no doesnt get to optimal value fast enough?

#Wasnt a problem with 1 hidden layer with batch size 2
# batch = 4, HL = 1, epochs = 100, test acc = 49%, loss = 1.31
# batch = 4, HL = 2, epochs = 100, test acc = 49%, loss = 1.29
# batch = 2, HL = 2, epochs = 100, test acc = 52%, loss = 1.10