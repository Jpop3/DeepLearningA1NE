import numpy as np
import pandas as pd
from MLP import *

input = np.load('Assignment1-Dataset/train_data.npy')
labels = np.load('Assignment1-Dataset/train_label.npy')
#print(labels.shape) #1 label for each vector of size 128. - Need to convert this to a vector of [0,0,0,1,0, ...]

#https://chat.openai.com/c/83234cc9-7d6b-43eb-8d9e-c87b68e5e125
def class_to_one_hot(class_label, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[class_label] = 1.
    return one_hot

def bin_class():
    #----------------------------------------------------------------------------------------
    #BASIC binary classifier
    # class_1 = np.hstack([np.random.normal( 1, 1, size=(25, 2)),  np.ones(shape=(25, 1))])
    # class_2 = np.hstack([np.random.normal(-1, 1, size=(25, 2)), -np.ones(shape=(25, 1))])
    # dataset = np.vstack([class_1, class_2])

    # nn = MLP([2,3,1], [None,'tanh','tanh'])
    # input_data = dataset[:,0:2]
    # output_data = dataset[:,2]
    # MSE = nn.fit(input_data, output_data, learning_rate=0.001, epochs=500)
    # print('loss:%f'%MSE[-1])
    #----------------------------------------------------------------------------------------
    pass

TRAINING_SIZE = 40000
# Set numpy seed for reproducibility
np.random.seed(0)

SETUP = {
    'epochs': 10,
    'lr': 0.001,
    'bn': True,
    'batch_size': 4,
    'hidden_layers': [128, 128, 64, 32, 10],
    'activations': [None, 'ReLU', 'ReLU', 'ReLU', 'softmax'],
    'input_size': 128
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

#Multi class classifer - fitting the neural network
# nn = MLP([128, 64, 32, 10], [None, 'ReLU', 'ReLU', 'softmax'], use_batch_norm=True) #can adjust hidden layers and activation functions

# Can pass None in for validation data if we dont want to use it
# CEL = nn.fit(input_training, labels_training, input_val, labels_val, learning_rate=0.001, epochs=100, batch_size=2) #can change parameters
# CEL = nn.fit(input_training, labels_training, input_val, labels_val, learning_rate=0.001, epochs=5, batch_size=1) #can change parameters
# print('Final training loss:%f'%CEL[-1])

nn = MLP(SETUP['hidden_layers'], SETUP['activations'], SETUP['bn'])
CEL = nn.fit(input_training, labels_training, input_val, labels_val, learning_rate=SETUP['lr'], epochs=SETUP['epochs'], batch_size=SETUP['batch_size'])


    ###### Results ######
PRINT_RESULTS = True

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
    print('\nConfusion matrix for train data:')
    print(confusion_matrix_train)
    print('\nConfusion matrix for test data:')
    print(confusion_matrix_test)


print(f'Train accuracy: {correct_train_count/TRAINING_SIZE} (count {correct_train_count} of {TRAINING_SIZE})') # train accuracy
print(f'Test accuracy: {correct_test_count/10000} (count {correct_test_count} of {10000})') # test accuracy

#Loss jumped back up?? after hitting 1.30 to 1.61 and more - this was because wasnt shuffling minibatches
#Internet said make the learning rate smaller. but no doesnt get to optimal value fast enough?
#https://www.analyticsvidhya.com/blog/2021/06/the-challenge-of-vanishing-exploding-gradients-in-deep-neural-networks/
#Wasnt a problem with 1 hidden layer with batch size 2
# batch = 4, HL = 1, epochs = 100, test acc = 49%, loss = 1.31
# batch = 4, HL = 2, epochs = 100, test acc = 49%, loss = 1.29
# batch = 2, HL = 2, epochs = 100, test acc = 52%, loss = 1.10

# Seed 0
# 'input_size': 128, 'hidden_layers': [128, 64, 32, 10], 'activations': [None, 'ReLU', 'ReLU', 'softmax'], 'bn': True,  'lr': 0.001, 'epochs': 5, 'batch_size': 2
# Epoch 1/5, Train loss: 2.12096, Val loss: 1.96318
# Epoch 2/5, Train loss: 1.84372, Val loss: 1.82901
# Epoch 3/5, Train loss: 1.74219, Val loss: 1.77627
# Epoch 4/5, Train loss: 1.69962, Val loss: 1.74475
# Epoch 5/5, Train loss: 1.64490, Val loss: 1.71292
# Train accuracy: 0.097425
# Test accuracy: 0.4067