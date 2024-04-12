import numpy as np
import pandas as pd
import pickle
from MLP import *
import sys
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

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

if len(sys.argv) < 2:
    with open("MLP_classifier.pkl", 'rb') as file:  
        model = pickle.load(file)
else:
    with open(sys.argv[1], 'rb') as file:  
        model = pickle.load(file)

output_test = model.predict(input_test)
# correct_test_count = np.sum(np.argmax(output_test, axis=1) == np.argmax(test_one_hot_labels, axis=1))
# Calulate the metrics for accuracy, precision, recall and f1 score
accuracy = accuracy_score(np.argmax(test_one_hot_labels, axis=1), np.argmax(output_test, axis=1))
precision = precision_score(np.argmax(test_one_hot_labels, axis=1), np.argmax(output_test, axis=1), average='macro')
recall = recall_score(np.argmax(test_one_hot_labels, axis=1), np.argmax(output_test, axis=1), average='macro')
f1 = f1_score(np.argmax(test_one_hot_labels, axis=1), np.argmax(output_test, axis=1), average='macro')
# Print f-string of final results
print(f'Test accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}')

# Confusion matrix for the test data
confusion_matrix_test = np.zeros((10,10))
for index, array in enumerate(output_test):
    predicted = np.argmax(array)
    actual = np.argmax(test_one_hot_labels[index])
    confusion_matrix_test[actual][predicted] += 1