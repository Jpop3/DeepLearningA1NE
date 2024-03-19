import numpy as np
import pandas as pd
from MLP import *

input = np.load('../Assignment1-Dataset/train_data.npy')

#print(input.shape) #50000 vectors of size 128.

labels = np.load('../Assignment1-Dataset/train_label.npy')
#print(labels.shape) #1 label for each vector of size 128. - Need to convert this to a vector of [0,0,0,1,0, ...]

#https://chat.openai.com/c/83234cc9-7d6b-43eb-8d9e-c87b68e5e125
def class_to_one_hot(class_label, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[class_label] = 1.
    return one_hot

#BASIC binary classifier
#I.e using the 128 input values per observation to determine the class

# class_1 = np.hstack([np.random.normal( 1, 1, size=(25, 2)),  np.ones(shape=(25, 1))])
# class_2 = np.hstack([np.random.normal(-1, 1, size=(25, 2)), -np.ones(shape=(25, 1))])
# dataset = np.vstack([class_1, class_2])

# nn = MLP([2,3,1], [None,'tanh','tanh'])
# input_data = dataset[:,0:2]
# output_data = dataset[:,2]
# MSE = nn.fit(input_data, output_data, learning_rate=0.001, epochs=500)
# print('loss:%f'%MSE[-1])


#I THINK THIS WORKS!!!!! WOOOOOOOO

#Training performance -
one_hot_labels = np.zeros((50000, 10))
for index, label in enumerate(labels):
    one_hot_labels[index] = class_to_one_hot(label, 10)

#Multi class classifer
nn = MLP([128, 64, 32, 10], [None, 'ReLU', 'ReLU', 'softmax'])
input_data = input
output_data = one_hot_labels
CEL = nn.fit(input_data, output_data, learning_rate=0.001, epochs=100, batch_size=2) 
print('loss:%f'%CEL[-1])

output = nn.predict(input_data)
correct_count = 0
for index, array in enumerate(output):
    if np.argmax(array) == np.argmax(one_hot_labels[index]):
        correct_count += 1
print(correct_count/50000) #58% correct classification! - ReLU - 64 hidden layer

#Test performance -
input_test = np.load('../Assignment1-Dataset/test_data.npy')
labels_test = np.load('../Assignment1-Dataset/test_label.npy')
test_one_hot_labels = np.zeros((10000, 10))
for index, label in enumerate(labels_test):
    test_one_hot_labels[index] = class_to_one_hot(label, 10)
output = nn.predict(input_test)
correct_count = 0
for index, array in enumerate(output):
    if np.argmax(array) == np.argmax(test_one_hot_labels[index]):
        correct_count += 1
print(correct_count/10000) #48% accuracy with tanh, lets go, 51% with ReLU - 64 hidden layer

#30 epochs, mini batch = 2, relu, training acc = , testing acc = 
#Loss jumped back up?? after hitting 1.30 to 1.61 and more
#Internet said make the learning rate smaller. but no doesnt get to optimal value fast enough?
#https://www.analyticsvidhya.com/blog/2021/06/the-challenge-of-vanishing-exploding-gradients-in-deep-neural-networks/
#Wasnt a problem with 1 hidden layer with batch size 2
# batch = 4, HL = 1, epochs = 100, test acc = 49%, loss = 1.31
# batch = 4, HL = 2, epochs = 100, test acc = 49%, loss = 1.29
# batch = 2, HL = 2, epochs = 100, test acc = 52%, loss = 1.10