import numpy as np
import pandas as pd
from MLP import *
import os

def class_to_one_hot(class_label, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[class_label] = 1.
    return one_hot

input_data = np.load('Assignment1-Dataset/train_data.npy')
labels = np.load('Assignment1-Dataset/train_label.npy')

one_hot_labels = np.array([class_to_one_hot(label, 10) for label in labels])

TRAINING_SIZE = 40000
np.random.seed(0)
indices = np.random.permutation(len(input_data))
training_idx, val_idx = indices[:TRAINING_SIZE], indices[TRAINING_SIZE:]
input_training, input_val = input_data[training_idx], input_data[val_idx]
labels_training, labels_val = one_hot_labels[training_idx], one_hot_labels[val_idx]

learning_rates = [0.001, 0.005, 0.01, 0.0001]
batch_sizes = [1, 2, 4, 8, 16, 32]
hidden_sizes = [
    [128, 96, 64, 10],
    [128, 64, 32, 10],
    [128, 128, 64, 10],
    [128, 64, 64, 10],
    [128, 96, 64, 32, 10],
    [128, 64, 32, 32, 10],
    [128, 128, 64, 32, 10],
    [128, 64, 64, 32, 10]
]
dropout_rates = [
    [0.05, 0.1, 0.2, 0],
    [0.05, 0.2, 0.2, 0],
    [0.05, 0.2, 0.4, 0],
    [0.05, 0.2, 0.5, 0],
    [0.05, 0.3, 0.3, 0],
    [0.1, 0.2, 0.3, 0],
    [0.1, 0.2, 0.4, 0],
    [0.1, 0.3, 0.4, 0],
    [0.1, 0.2, 0.5, 0],
]
bns = [True, False]
weight_decays = [0, 1e-5, 1e-7]
optimisers = ['Adam', None]

# Initialize or clear the file where results will be saved
results_file = 'hyperparameter_tuning_results_2.csv'
# Check if the file exists and if it doesn't create it with the header
if not os.path.exists(results_file):
    with open(results_file, 'w') as file:
        file.write("lr,batch_size,dropout_rate,train_accuracy,val_accuracy\n")

# for lr in learning_rates:
#     for batch_size in batch_sizes:
#         for dropout_rate in dropout_rates:
#             SETUP = {
#                 'epochs': 30,
#                 'lr': lr,
#                 'bn': True,
#                 'batch_size': batch_size,
#                 'dropout_rate': [dropout_rate] * 3 + [0],  # Example simplification
#                 'hidden_layers': [128, 64, 32, 10],
#                 'activations': [None, 'ReLU', 'ReLU', 'softmax'],
#                 'input_size': 128,
#                 'weight_decay': 0,
#                 'optimiser': None
#             }
#             # Print the current configuration
#             print(f"Training with lr={lr}, batch_size={batch_size}, dropout_rate={dropout_rate}")
#             nn = MLP(SETUP['hidden_layers'], SETUP['activations'], SETUP['bn'], SETUP['weight_decay'], SETUP['dropout_rate'])
#             nn.fit(input_training, labels_training, input_val, labels_val, learning_rate=SETUP['lr'], epochs=SETUP['epochs'], batch_size=SETUP['batch_size'], optimiser=SETUP['optimiser'])

#             train_accuracy = np.mean(np.argmax(nn.predict(input_training), axis=1) == np.argmax(labels_training, axis=1))
#             val_accuracy = np.mean(np.argmax(nn.predict(input_val), axis=1) == np.argmax(labels_val, axis=1))

#             # Print the results
#             print(f"Training accuracy: {train_accuracy:.5f}, Validation accuracy: {val_accuracy:.5f}\n")
            
#             # Write results to file after each configuration is evaluated
#             with open(results_file, 'a') as file:
#                 file.write(f"{lr},{batch_size},{dropout_rate},{train_accuracy},{val_accuracy}\n")

print("Hyperparameter tuning completed. Results saved incrementally to 'hyperparameter_tuning_results.csv'.")
