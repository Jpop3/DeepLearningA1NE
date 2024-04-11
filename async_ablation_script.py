import numpy as np
from MLP import *
import os
import time
import itertools
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import csv
import concurrent.futures
import json
import pickle
import traceback
import sys

def generate_folds(n_folds=10):
    
    X = np.load('Assignment1-Dataset/train_data.npy')
    labels = np.load('Assignment1-Dataset/train_label.npy')
    y = np.array([MLP.class_to_one_hot(label, 10) for label in labels])

    # Seed for reproducibility
    np.random.seed(0)

    # 5-fold Cross Validation
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[:n_samples % n_folds] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop

    # Generate training and validation datasets for each fold
    fold_splits = []
    for i in range(n_folds):
            # Generate training and validation sets for this fold
            val_indices = folds[i]
            train_indices = np.hstack(folds[:i] + folds[i+1:])
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            fold_splits.append((X_train, y_train, X_val, y_val))
            
    # Make copies of the fold_splits data for async processing
    fold_splits = [(X_train.copy(), y_train.copy(), X_val.copy(), y_val.copy()) for (X_train, y_train, X_val, y_val) in fold_splits]
    return fold_splits

# fold_splits = fold_splits[:1]

def can_be_pickled(obj):
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError) as e:
        print(f"Cannot be pickled: {e}")
        return False

def log_detailed_metrics(hyperparams_dict, cv_metrics, filename="detailed_model_abalation.csv"):
    # Check if file exists, if not, write headers
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Fold", "Hyperparameters", "Train Losses", "Val Losses", "Val Accuracy", "Val Precision", "Val Recall", "Val F1", "Time Taken", "Early Stop Epoch"])
        if cv_metrics is not None:
            for i, metrics in enumerate(cv_metrics):
                # Convert ndarray to list if necessary
                detailed_train_loss = metrics['detailed_train_loss']
                detailed_val_loss = metrics['detailed_val_loss']
                
                if isinstance(detailed_train_loss, np.ndarray):
                    detailed_train_loss = detailed_train_loss.tolist()
                if isinstance(detailed_val_loss, np.ndarray):
                    detailed_val_loss = detailed_val_loss.tolist()
                    
                # Serialize lists (or other JSON-serializable structures)
                detailed_train_loss_str = json.dumps(detailed_train_loss)
                detailed_val_loss_str = json.dumps(detailed_val_loss)
                
                writer.writerow([
                    i + 1,
                    json.dumps(hyperparams_dict),
                    detailed_train_loss_str,
                    detailed_val_loss_str,
                    metrics['val_accuracy'],
                    metrics['val_precision'],
                    metrics['val_recall'],
                    metrics['val_f1'],
                    metrics['time_taken'],
                    metrics['early_stop_epoch']
                ])
        else:
            writer.writerow([-1, json.dumps(hyperparams_dict), None, None, None, None, None, None, None, None])

def train_and_evaluate_fold(X_train, y_train, X_val, y_val, hyperparams):
    """
    Trains the model for one fold and evaluates it.
    This function is designed to be run in a separate process.
    """
    try:
        # print('Starting a fold with hyperparameters:', hyperparams_dict)
        # Initialize MLP model with current hyperparameters
        nn = MLP(hyperparams['hidden_layers'], hyperparams['activations'], hyperparams['bn'], hyperparams['weight_decay'], hyperparams['dropout_rate'])

        # Fit the model
        start = time.time()
        nn_output = nn.fit(X_train, y_train, X_val, y_val, learning_rate=hyperparams['lr'], epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'], optimiser=hyperparams['optimiser'], early_stopping=hyperparams['early_stopping'], verbose=False)
        end = time.time()

        # Extract metrics
        train_loss, val_loss, early_stop_epoch = nn_output
        time_taken = end - start

        # Predict on validation set
        output_val = nn.predict(X_val)

        # Calculate metrics
        val_accuracy = accuracy_score(np.argmax(y_val, axis=1), np.argmax(output_val, axis=1))
        val_precision = precision_score(np.argmax(y_val, axis=1), np.argmax(output_val, axis=1), average='macro')
        val_recall = recall_score(np.argmax(y_val, axis=1), np.argmax(output_val, axis=1), average='macro')
        val_f1 = f1_score(np.argmax(y_val, axis=1), np.argmax(output_val, axis=1), average='macro')
        
        # Print metrics
        print(f"Train Loss: {train_loss[-1]}, Val Loss: {val_loss[-1]}, Val Accuracy: {val_accuracy}, Val Precision: {val_precision}, Val Recall: {val_recall}, Val F1: {val_f1}, Time Taken: {time_taken}")

        return {
            'train_loss': train_loss[-1],
            'val_loss': val_loss[-1],
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'time_taken': time_taken,
            'early_stop_epoch': early_stop_epoch,
            'detailed_train_loss': train_loss,
            'detailed_val_loss': val_loss
        }
    except Exception as e:
        with open('process_log.txt', "a") as log_file:
            log_file.write(f"Error: {e}\n")
        return {
            'train_loss': None,
            'val_loss': None,
            'val_accuracy': None,
            'val_precision': None,
            'val_recall': None,
            'val_f1': None,
            'time_taken': None,
            'early_stop_epoch': None,
            'detailed_train_loss': None,
            'detailed_val_loss': None
        }
        

# Function to run your model CV asynchronously
def run_model_cv_async(fold_splits, hyperparams):
    results = []
            
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = []
        for X_train, y_train, X_val, y_val in fold_splits:
            future = executor.submit(train_and_evaluate_fold, X_train, y_train, X_val, y_val, hyperparams)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    return results


if __name__ == '__main__':
    
    # Get baseline config file
    if len(sys.argv) < 5:
        print("(1) config file (2) ablation options file (3) log file path (4) detailed fold file path. Exiting...")
        exit()
    else:
        with open(sys.argv[1]) as f:
            config = json.load(f)
        with open(sys.argv[2]) as f:
            ablation_dict = json.load(f)
            ablation_dict = ablation_dict['ablation_dict']
        log_filepath = sys.argv[3]
        detailed_log_filepath = sys.argv[4]
    
    ### Constants ###
    # SETUP = {
    #     'epochs': 50,
    #     'activations': [None, 'ReLU', 'softmax'],
    #     'input_size': 128,
    #     'early_stopping': (10, 0.001)
    # }

    ### Options for Hyper-Parameter Abalation ###
    # lr_options = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    # hidden_layer_options = [[128, 64, 32, 10], [128, 96, 64, 10]]
    
    # batch_size_options = [1, 2, 8, 16, 32, 64]
    # weight_decay_options = [0.001, 0.001, 0.005, 0.01]
    # drop_out_options = [[0, 0.1, 0], [0.05, 0.2, 0], [0.1, 0.3, 0], [0.2, 0.4, 0]]
    # optimiser_options = ['Adam', 'Momentum']
    # bn_option = [1]
    # activations_options = [["None", "logistic", "softmax"], ["None", "tanh", "softmax"]]
    
    # ablation_dict = {
    #     'batch_size': [1, 2, 8, 16, 32, 64],
    #     'weight_decay': [0.001, 0.001, 0.005, 0.01],
    #     'dropout_rate': [[0, 0.1, 0], [0.05, 0.2, 0], [0.1, 0.3, 0], [0.2, 0.4, 0]],
    #     'optimiser': ['Adam', 'Momentum'],
    #     'bn': [1],
    #     'activations': [["None", "logistic", "softmax"], ["None", "tanh", "softmax"]]
    # }
    
    fold_splits = generate_folds()
    
    # Configurations for abalation, including the baseline
    abalation_hyperparams_list = [config['SETUP'].copy()]
    
    # Generate list of all hyperparameter combinations using abalation_dict
    for key, value in ablation_dict.items():
        for val in value:
            temp_dict = config['SETUP'].copy()
            temp_dict[key] = val
            
            # Number of hidden layer adjustments
            if key == 'hidden_layers':
                if len(val) == 4:
                    temp_dict['activations'] = ["None", "ReLU", "ReLU", "softmax"]
                    temp_dict['dropout_rate'] = [0.0, 0.0, 0.0, 0.0]
                elif len(val) == 5:
                    temp_dict['activations'] = ["None", "ReLU", "ReLU", "ReLU", "softmax"]
                    temp_dict['dropout_rate'] = [0.0, 0.0, 0.0, 0.0, 0.0]
                    
            abalation_hyperparams_list.append(temp_dict)
    
    for i in abalation_hyperparams_list:
        print(i)
    
    
    # Continue from flag
    # begin_from_here = True # Set to true to start from beginning, false to start from the from_dict
    # from_dict = {'weight_decay': 0.0, 'hidden_layers': [128, 64, 32, 10], 'dropout': [0.0, 0.0, 0.0, 0.0], 'lr': 0.01, 'optimiser': None, 'bn': False, 'batch_size': 2}
    
    # Open a file to write the output
    with open(log_filepath, 'a') as f:
        print('Begin Hyper-parameter Tuning -')
        for hyperparams in abalation_hyperparams_list:
            
            # Continue Mechanism
            # if begin_from_here == False:
            #     if hyperparams_dict == from_dict:
            #         begin_from_here = True
            #     else:
            #         continue

            print(f'\n## Running model {str(hyperparams)}')
            print(f'\n## Running model {str(hyperparams)}', file=f, flush=True)
            # print(can_be_pickled(hyperparams_dict))
            try:
                start = time.time()
                # Use the asynchronous version here
                cv_metrics = run_model_cv_async(fold_splits, hyperparams)
                end = time.time()
                time_taken = end - start
                
                print(f'## Results ({time_taken}s) avg val accuracy: {np.mean([m["val_accuracy"] for m in cv_metrics])} - {str(hyperparams)}')
                print(f'## Results ({time_taken}s) avg val accuracy: {np.mean([m["val_accuracy"] for m in cv_metrics])}', file=f, flush=True)
                
                log_detailed_metrics(hyperparams, cv_metrics, detailed_log_filepath)

            except Exception as e:
                print(f'Error: {e}')
                # traceback_str = traceback.format_exc()  # This gives you the full traceback as a string
                # print(f'Error: {traceback_str}')
                try:
                    log_detailed_metrics(hyperparams, None, detailed_log_filepath)  # Log a dummy value in case of error
                except:
                    pass