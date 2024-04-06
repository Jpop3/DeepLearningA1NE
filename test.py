import numpy as np
import pandas as pd
from MLP import *

# x = np.array([[1,2,3], [3,4,5], [6,7,8]])
# y = np.array(([1,0], [0,1], [0,-1]))
# p = np.random.permutation(len(x))
# print(x[p], y[p])

# print(sum(x)/len(x))

input = np.load('Assignment1-Dataset/train_data.npy')

#print(input.shape) #50000 vectors of size 128.

labels = np.load('Assignment1-Dataset/train_label.npy')
#print(labels.shape) #1 label for each vector of size 128. - Need to convert this to a vector of [0,0,0,1,0, ...]

#https://chat.openai.com/c/83234cc9-7d6b-43eb-8d9e-c87b68e5e125
def class_to_one_hot(class_label, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[class_label] = 1.
    return one_hot

#Make one hot labels
one_hot_labels = np.zeros((50000, 10))
for index, label in enumerate(labels):
    one_hot_labels[index] = class_to_one_hot(label, 10)

# print(input)
# print(input.shape)

# print(labels)
# print(labels.shape)

# print(one_hot_labels)
# print(one_hot_labels.shape)

x = np.array([0.0, 4.0, 6.0])
print(x * 0.98)