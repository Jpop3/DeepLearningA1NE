import numpy as np
from MLP import *
import os
import time
import itertools
from sklearn.metrics import accuracy_score, recall_score, f1_score
import csv
import concurrent.futures
import json
import pickle
import traceback
import sys

def generate_folds():
    X = np.load('Assignment1-Dataset/train_data.npy')
    labels = np.load('Assignment1-Dataset/train_label.npy')
    y = np.array([MLP.class_to_one_hot(label, 10) for label in labels])

    # Seed for reproducibility
    np.random.seed(0)

    # 5-fold Cross Validation
    n_samples = X.shape[0]
    n_folds = 5
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

def log_detailed_metrics(hyperparams_dict, cv_metrics, filename="detailed_model_performance.csv"):
    # Check if file exists, if not, write headers
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Fold", "Hyperparameters", "Train Losses", "Val Losses", "Val Accuracy", "Val Recall", "Val F1", "Time Taken", "Early Stop Epoch"])
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
                    metrics['val_recall'],
                    metrics['val_f1'],
                    metrics['time_taken'],
                    metrics['early_stop_epoch']
                ])
        else:
            writer.writerow([-1, json.dumps(hyperparams_dict), None, None, None, None, None, None, None])

def train_and_evaluate_fold(X_train, y_train, X_val, y_val, hyperparams_dict, static_params):
    """
    Trains the model for one fold and evaluates it.
    This function is designed to be run in a separate process.
    """
    try:
        
        hidden_layers = hyperparams_dict['hidden_layers']
        lr = hyperparams_dict['lr']
        batch_size = hyperparams_dict['batch_size']
        optimiser = hyperparams_dict['optimiser']
        bn = hyperparams_dict['bn']
        weight_decay = hyperparams_dict['weight_decay']
        dropout = hyperparams_dict['dropout']
        
        # print('Starting a fold with hyperparameters:', hyperparams_dict)
        # Initialize MLP model with current hyperparameters
        nn = MLP(hidden_layers, static_params['activations'], bn, weight_decay, dropout)

        # Fit the model
        start = time.time()
        nn_output = nn.fit(X_train, y_train, X_val, y_val, learning_rate=lr, epochs=static_params['epochs'], batch_size=batch_size, optimiser=optimiser, early_stopping=static_params['early_stopping'])
        end = time.time()

        # Extract metrics
        train_loss, val_loss, early_stop_epoch = nn_output
        time_taken = end - start

        # Predict on validation set
        output_val = nn.predict(X_val)

        # Calculate metrics
        val_accuracy = accuracy_score(np.argmax(y_val, axis=1), np.argmax(output_val, axis=1))
        val_recall = recall_score(np.argmax(y_val, axis=1), np.argmax(output_val, axis=1), average='macro')
        val_f1 = f1_score(np.argmax(y_val, axis=1), np.argmax(output_val, axis=1), average='macro')
        
        # Print metrics
        print(f"Train Loss: {train_loss[-1]}, Val Loss: {val_loss[-1]}, Val Accuracy: {val_accuracy}, Val Recall: {val_recall}, Val F1: {val_f1}, Time Taken: {time_taken}")

        return {
            'train_loss': train_loss[-1],
            'val_loss': val_loss[-1],
            'val_accuracy': val_accuracy,
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
            'val_recall': None,
            'val_f1': None,
            'time_taken': None,
            'early_stop_epoch': None,
            'detailed_train_loss': None,
            'detailed_val_loss': None
        }
        

# Function to run your model CV asynchronously
def run_model_cv_async(fold_splits, hyperparams_dict, static_params):
    results = []
    # with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    #     # Schedule the execution of each fold
    #     print('Sending worker')
    #     futures = [executor.submit(train_and_evaluate_fold, X_train, y_train, X_val, y_val, hyperparams_dict)
    #                for (X_train, y_train, X_val, y_val) in fold_splits]

    #     for future in concurrent.futures.as_completed(futures):
    #         results.append(future.result())
            
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = []
        for X_train, y_train, X_val, y_val in fold_splits:
            # print('Sending worker')
            # print(f'X_train: {can_be_pickled(X_train)}, y_train: {can_be_pickled(y_train)}, X_val: {can_be_pickled(X_val)}, y_val: {can_be_pickled(y_val)}, hyperparams_dict: {can_be_pickled(hyperparams_dict)}, static_params: {can_be_pickled(static_params)}')
            # Correctly pass the slices of your dataset and hyperparameters to the function
            future = executor.submit(train_and_evaluate_fold, X_train, y_train, X_val, y_val, hyperparams_dict, static_params)
            futures.append(future)
            # print('Worker sent')

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            # print('Worker finished')

    return results
    
    print('Running in serial mode')
    for (X_train, y_train, X_val, y_val) in fold_splits:
        results.append(train_and_evaluate_fold(X_train, y_train, X_val, y_val, hyperparams_dict))

    # After all futures are completed, results are collected
    # Now, you can aggregate or process results as needed
    return results



if __name__ == '__main__':
    # Redirect all system outputs to a log file
    # sys.stdout = open('process_log.txt', 'w')
    
    ### Constants ###
    SETUP = {
        'epochs': 50,
        'activations': [None, 'ReLU', 'ReLU', 'softmax'],
        'input_size': 128,
        'early_stopping': (10, 0.001)
    }

    ### Options for Hyper-Parameters ###
    weight_decay_options = [0.0, 0.0001, 0.001]
    hidden_layer_options = [[128, 64, 32, 10], [128, 96, 64, 10]]
    drop_out_options = [[0.0, 0.0, 0.0, 0.0], [0.05, 0.2, 0.2, 0.0], [0.1, 0.3, 0.3, 0.1]]
    lr_options = [0.001, 0.0001, 0.01]
    optimiser_options = [None, 'Adam', 'Momentum']
    bn_option = [False, True]
    batch_size_options = [16,8,2]
    
    fold_splits = generate_folds()
    
    # Continue from flag
    begin_from_here = False # Set to true to start from beginning
    # from_dict = {'weight_decay': 0.0, 'hidden_layers': [128, 64, 32, 10], 'dropout': [0.0, 0.0, 0.0, 0.0], 'lr': 0.01, 'optimiser': None, 'bn': False, 'batch_size': 2}
    from_dict = {'weight_decay': 0.0, 'hidden_layers': [128, 96, 64, 10], 'dropout': [0.5, 0.2, 0.2, 0.0], 'lr': 0.001, 'optimiser': 'Adam', 'bn': True, 'batch_size': 16}
    
    # Open a file to write the output
    with open('output_log.txt', 'w') as f:
        # sys.stdout = f  # Set stdout to the file object

        print('Begin Hyper-parameter Tuning -')
        for hyperparams in itertools.product(weight_decay_options, hidden_layer_options, drop_out_options, lr_options, optimiser_options, bn_option, batch_size_options):
            hyperparams_dict = {
                'weight_decay': hyperparams[0],
                'hidden_layers': hyperparams[1],
                'dropout': hyperparams[2],
                'lr': hyperparams[3],
                'optimiser': hyperparams[4],
                'bn': hyperparams[5],
                'batch_size': hyperparams[6]
            }
            
            # Continue Mechanism
            if begin_from_here == False:
                if hyperparams_dict == from_dict:
                    begin_from_here = True
                else:
                    continue

            print(f'\n### Running model {str(hyperparams_dict)}')
            print(f'\n### Running model {str(hyperparams_dict)}', file=f, flush=True)
            # print(can_be_pickled(hyperparams_dict))
            try:
                start = time.time()
                # Use the asynchronous version here
                cv_metrics = run_model_cv_async(fold_splits, hyperparams_dict, SETUP)
                end = time.time()
                time_taken = end - start
                
                print(f'## Model {str(hyperparams_dict)} took {time_taken} seconds: avg val accuracy: {np.mean([m["val_accuracy"] for m in cv_metrics])}')
                print(f'## Model {str(hyperparams_dict)} took {time_taken} seconds: avg val accuracy: {np.mean([m["val_accuracy"] for m in cv_metrics])}', file=f, flush=True)
                
                log_detailed_metrics(hyperparams_dict, cv_metrics)

            except Exception as e:
                print(f'Error: {e}')
                # traceback_str = traceback.format_exc()  # This gives you the full traceback as a string
                # print(f'Error: {traceback_str}')
                try:
                    log_detailed_metrics(hyperparams_dict, None, 99999)  # Log a dummy value in case of error
                except:
                    pass

# def parallel_hyperparam_search(hyperparam_combinations, fold_splits, static_params):
#     """
#     Function to run multiple instances of run_model_cv_async in parallel.
#     """
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         futures = []
#         for hyperparams in hyperparam_combinations:
#             hyperparams_dict = {
#                 'weight_decay': hyperparams[0],
#                 'hidden_layers': hyperparams[1],
#                 'dropout': hyperparams[2],
#                 'lr': hyperparams[3],
#                 'optimiser': hyperparams[4],
#                 'bn': hyperparams[5],
#                 'batch_size': hyperparams[6]
#             }
#             future = executor.submit(run_model_cv_async, fold_splits, hyperparams_dict, static_params)
#             futures.append(future)
        
#         for future in concurrent.futures.as_completed(futures):
#             cv_metrics = future.result()
#             # Process and log the cv_metrics as needed for each hyperparameter combination
#             log_detailed_metrics(hyperparams_dict, cv_metrics)
            
# if __name__ == '__main__':
    
#     ### Constants ###
#     SETUP = {
#         'epochs': 50,
#         'activations': [None, 'ReLU', 'ReLU', 'softmax'],
#         'input_size': 128,
#         'early_stopping': (10, 0.001)
#     }

#     ### Options for Hyper-Parameters ###
#     weight_decay_options = [0.0, 0.0001, 0.001]
#     hidden_layer_options = [[128, 64, 32, 10], [128, 64, 32, 10]]
#     drop_out_options = [[0.0, 0.0, 0.0, 0.0], [0.5, 0.2, 0.2, 0.0], [0.1, 0.3, 0.3, 0.1]]
#     lr_options = [0.001, 0.0001, 0.01]
#     optimiser_options = [None, 'Adam', 'Momentum']
#     bn_option = [False, True]
#     batch_size_options = [16,8,2]
#     fold_splits = generate_folds()
    
#     # Generate all combinations of hyperparameters
#     hyperparam_combinations = list(itertools.product(weight_decay_options, hidden_layer_options, drop_out_options, lr_options, optimiser_options, bn_option, batch_size_options))
    
#     # Split hyperparameter combinations for parallel execution
#     # Example: Splitting the list into two for demonstration
#     split_index = len(hyperparam_combinations) // 2
#     combinations_part1 = hyperparam_combinations[:split_index]
#     combinations_part2 = hyperparam_combinations[split_index:]

#     # Run in parallel
#     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#         future1 = executor.submit(parallel_hyperparam_search, combinations_part1, fold_splits, SETUP)
#         future2 = executor.submit(parallel_hyperparam_search, combinations_part2, fold_splits, SETUP)
        
#         # Wait for both futures to complete
#         concurrent.futures.wait([future1, future2], return_when=concurrent.futures.ALL_COMPLETED)
        
#         print("All hyperparameter searches completed.")
