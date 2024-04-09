import numpy as np
import pandas as pd
from MLP import *
import matplotlib.pyplot as plt
import time


input = np.load('Assignment1-Dataset/train_data.npy')
labels = np.load('Assignment1-Dataset/train_label.npy')

#https://chat.openai.com/c/83234cc9-7d6b-43eb-8d9e-c87b68e5e125
def class_to_one_hot(class_label, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[class_label] = 1.
    return one_hot


TRAINING_SIZE = 40000
# Set numpy seed for reproducibility
np.random.seed(0)

SETUP = {
    'epochs': 5,
    'lr': 0.001,
    'bn': False,
    'batch_size': 32,
    'dropout_rate': [0, 0.2, 0.1, 0], # dropout rate for each layer: eg. [0.1, 0.2, 0.4, 0]
    'hidden_layers': [128, 64, 32, 10],
    'activations': [None, 'ReLU', 'ReLU', 'softmax'],
    'input_size': 128,
    'weight_decay': 0,
    'optimiser': 'Adam',
    'early_stopping': (10, 0.001)
}
#Issue, how to allow users to specific params for adams, momentum etc

#Make one hot labels
one_hot_labels = np.zeros((50000, 10))
for index, label in enumerate(labels):
    one_hot_labels[index] = class_to_one_hot(label, 10)

## Create training and validation sets from randomly shuffled data
indices = np.random.permutation(input.shape[0])
training_idx, val_idx = indices[:TRAINING_SIZE], indices[TRAINING_SIZE:]
input_training, input_val = input[training_idx,:], input[val_idx,:]
labels_training, labels_val = one_hot_labels[training_idx,:], one_hot_labels[val_idx,:]


# print(f'Training indices: {training_idx[:10]}, Validation indices: {val_idx[:10]}')
# print(f'Shapes: input_training: {input_training.shape}, labels_training: {labels_training.shape}, input_val: {input_val.shape}, labels_val: {labels_val.shape}')

input_training = np.array(input_training)
labels_training = np.array(labels_training)
input_val = np.array(input_val)
labels_val = np.array(labels_val)

##### Model #####
nn = MLP(SETUP['hidden_layers'], SETUP['activations'], SETUP['bn'], SETUP['weight_decay'], SETUP['dropout_rate'])

# Start timer for training
start = time.time()
print('Training model...')
CEL = nn.fit(input_training, labels_training, input_val, labels_val, learning_rate=SETUP['lr'], epochs=SETUP['epochs'], batch_size=SETUP['batch_size'], optimiser=SETUP['optimiser'], early_stopping=SETUP['early_stopping'])

# End timer for training
end = time.time()
print(f'Training took {end - start} seconds')

print(CEL)


###### Results ######
PRINT_RESULTS = True
PRINT_CONFUSION_MATRIX = False

### Training performance ###
# Training accuracy
output_train = nn.predict(input_training)
correct_train_count = 0
for index, array in enumerate(output_train):
    if np.argmax(array) == np.argmax(labels_training[index]): #the largest val in the vector indicates what class its predicting.
        correct_train_count += 1
# print('Train accuracy:', correct_train_count/TRAINING_SIZE) #train accuracy

# Confusion matrix for the training data
confusion_matrix_train = np.zeros((10,10))
for index, array in enumerate(output_train):
    predicted = np.argmax(array)
    actual = np.argmax(labels_training[index])
    confusion_matrix_train[actual][predicted] += 1

### Test performance ###
input_test = np.load('Assignment1-Dataset/test_data.npy') #Load datasets
labels_test = np.load('Assignment1-Dataset/test_label.npy') #Load datasets
test_one_hot_labels = np.zeros((10000, 10)) 
for index, label in enumerate(labels_test):
    test_one_hot_labels[index] = class_to_one_hot(label, 10) #Make one hot labels

# Test accuracy
output_test = nn.predict(input_test)
correct_test_count = 0
for index, array in enumerate(output_test):
    if np.argmax(array) == np.argmax(test_one_hot_labels[index]):
        correct_test_count += 1
# print('Test accuracy:', correct_test_count/10000) #test accuracy

# Confusion matrix for the test data
confusion_matrix_test = np.zeros((10,10))
for index, array in enumerate(output_test):
    predicted = np.argmax(array)
    actual = np.argmax(test_one_hot_labels[index])
    confusion_matrix_test[actual][predicted] += 1
   
   
if PRINT_RESULTS:
    print('Results from setup:')
    print(SETUP)
    # Print confusion matrix as a table and integers
    if PRINT_CONFUSION_MATRIX:
        print('\nConfusion matrix for train data:')
        print(pd.DataFrame(confusion_matrix_train.astype(int)))
        print('\nConfusion matrix for test data:')
        print(pd.DataFrame(confusion_matrix_test.astype(int)))


print(f'Train accuracy: {correct_train_count/TRAINING_SIZE} (count {correct_train_count} of {TRAINING_SIZE})') # train accuracy
print(f'Test accuracy: {correct_test_count/10000} (count {correct_test_count} of {10000})') # test accuracy
