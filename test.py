import numpy as np
import pandas as pd
import pickle
from MLP import *

#https://chat.openai.com/c/83234cc9-7d6b-43eb-8d9e-c87b68e5e125
def class_to_one_hot(class_label, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[class_label] = 1.
    return one_hot

input_test = np.load('Assignment1-Dataset/test_data.npy') #Load datasets
labels_test = np.load('Assignment1-Dataset/test_label.npy') #Load datasets
test_one_hot_labels = np.zeros((10000, 10)) 
for index, label in enumerate(labels_test):
    test_one_hot_labels[index] = class_to_one_hot(label, 10) #Make one hot labels

with open("MLP_classifier.pkl", 'rb') as file:  
    model = pickle.load(file)

output_test = model.predict(input_test)
correct_test_count = 0
for index, array in enumerate(output_test):
    if np.argmax(array) == np.argmax(test_one_hot_labels[index]):
        correct_test_count += 1
print('Test accuracy:', correct_test_count/10000) #test accuracy

# Confusion matrix for the test data
confusion_matrix_test = np.zeros((10,10))
for index, array in enumerate(output_test):
    predicted = np.argmax(array)
    actual = np.argmax(test_one_hot_labels[index])
    confusion_matrix_test[actual][predicted] += 1