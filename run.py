import numpy as np
import pandas as pd
from MLP import *
import matplotlib.pyplot as plt


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
    'epochs': 50,
    'lr': 0.001,
    'bn': True,
    'batch_size': 2,
    'dropout_rate': [0.05, 0.3, 0.2, 0], # dropout rate for each layer: eg. [0.1, 0.2, 0.4, 0]
    'hidden_layers': [128, 96, 64, 10],
    'activations': [None, 'ReLU', 'ReLU', 'softmax'],
    'input_size': 128,
    'weight_decay': 0, # 1e-5 1e-7
    'optimiser': 'Adam'
}

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
CEL = nn.fit(input_training, labels_training, input_val, labels_val, learning_rate=SETUP['lr'], epochs=SETUP['epochs'], batch_size=SETUP['batch_size'], optimiser=SETUP['optimiser'])

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
    print('Setup:', SETUP)
    # Print confusion matrix as a table and integers
    if PRINT_CONFUSION_MATRIX:
        print('\nConfusion matrix for train data:')
        print(pd.DataFrame(confusion_matrix_train.astype(int)))
        print('\nConfusion matrix for test data:')
        print(pd.DataFrame(confusion_matrix_test.astype(int)))


print(f'Train accuracy: {correct_train_count/TRAINING_SIZE} (count {correct_train_count} of {TRAINING_SIZE})') # train accuracy
print(f'Test accuracy: {correct_test_count/10000} (count {correct_test_count} of {10000})') # test accuracy


# Print a training and validation loss graph
plt.plot(CEL[0], label='Train loss')
plt.plot(CEL[1], label='Val loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# Seed 0
# 'input_size': 128, 'hidden_layers': [128, 64, 32, 10], 'activations': [None, 'ReLU', 'ReLU', 'softmax'], 'bn': True,  'lr': 0.001, 'epochs': 5, 'batch_size': 2
# Epoch 1/5, Train loss: 2.12096, Val loss: 1.96318
# Epoch 2/5, Train loss: 1.84372, Val loss: 1.82901
# Epoch 3/5, Train loss: 1.74219, Val loss: 1.77627
# Epoch 4/5, Train loss: 1.69962, Val loss: 1.74475
# Epoch 5/5, Train loss: 1.64490, Val loss: 1.71292
# Train accuracy: 0.097425 - this is incorrect
# Test accuracy: 0.4067

#Seed 0
# 'input_size': 128, 'hidden_layers': [128, 64, 32, 10], 'activations': [None, 'ReLU', 'ReLU', 'softmax'], 'bn': False,  'lr': 0.001, 'epochs': 5, 'batch_size': 2
# Epoch 1/5, Train loss: 2.05673, Val loss: 1.90639
# Epoch 2/5, Train loss: 1.80091, Val loss: 1.78973
# Epoch 3/5, Train loss: 1.71268, Val loss: 1.73610
# Epoch 4/5, Train loss: 1.66942, Val loss: 1.70069
# Epoch 5/5, Train loss: 1.61815, Val loss: 1.68146

#Seed 0
# Epoch 1/5, Train loss: 1.72801, Val loss: 1.78680
# Epoch 2/5, Train loss: 1.53476, Val loss: 1.64534
# Epoch 3/5, Train loss: 1.44643, Val loss: 1.60790
# Epoch 4/5, Train loss: 1.42013, Val loss: 1.59447
# Epoch 5/5, Train loss: 1.37113, Val loss: 1.57239
# Setup: {'epochs': 5, 'lr': 0.01, 'bn': False, 'batch_size': 2, 'hidden_layers': [128, 64, 32, 10], 'activations': [None, 'ReLU', 'ReLU', 'softmax'], 'input_size': 128}
# Train accuracy: 0.4586 (count 18344 of 40000)
# Test accuracy: 0.446 (count 4460 of 10000)