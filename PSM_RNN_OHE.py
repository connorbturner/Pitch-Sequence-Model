### RNN Model for Conditional Pitch Sequencing; One-Hot Encoded Sequence and Context
### Author: Connor Turner

### Suppress irrelevant messages in output:

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

### Import relevant modules and functions:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, optimizers
from keras.losses import CategoricalCrossentropy


### Import the data

model_training_data = pd.read_csv('rnn_ohe_model_data.csv')
model_training_data = model_training_data[model_training_data['game_year'] == 2021]


### Build helper function to pre-process sequence data:

def sequence_builder(sequence, num_pitches):
    """
    Helper function that takes a sequence of pitch data, splits it into n(n+1)/2 sub-sequences, and
    builds a new set of sequence arrays, context vectors, and target vectors for each sub-sequence

    Inputs:
    - sequence (np.ndarray): Matrix of pitch data for a given at-bat

    Outputs:
    - new_sequences (np.ndarray): 3D Array of pitch sequence data for each sub-sequence
    - context_values (np.ndarray): Context vectors for each sub-sequence
    - y_values (np.ndarray): Target vectors for each sub-sequence
    """

    n = len(sequence)                                                    # Number of pitches in at-bat
    new_sequences = np.full((n, 8, 34 + num_pitches), -1)                # Initialize sequence array
    context_values = np.zeros((n, 37 + (2 * num_pitches)))               # Initialize context vectors
    y_values = np.zeros((n, num_pitches))                                # Initialize target vectors

    idx_1 = 34 + num_pitches
    idx_2 = 14 + num_pitches
    idx_3 = idx_2 + 37 + (2 * num_pitches)

    for i in range(n):
        new_sequences[i,0:i,:] = sequence[0:i, 0:idx_1]                  # Record the sequence data
        context_values[i,:] = sequence[i, idx_2:idx_3]                   # Record the context data
        y_values[i,:] = sequence[i, :num_pitches]                        # Record the target vector

    return new_sequences, context_values, y_values


### Initialize vectors to record model performance:

pitcher_ids = np.unique(model_training_data['pitcher_id'])[:30]
pitcher_names = model_training_data[model_training_data['pitcher_id'].isin(pitcher_ids)]['pitcher_name'].unique()
sample_size = np.zeros((len(pitcher_ids),))
training_accuracies = np.zeros((len(pitcher_ids),))
testing_accuracies = np.zeros((len(pitcher_ids),))
best_epochs = np.zeros((len(pitcher_ids),))
arsenal_size = np.zeros((len(pitcher_ids),))


### Train model for each pitcher:

for pitcher in pitcher_ids:
    
    ### Load data:

    print(f"\n{pitcher_names[np.where(pitcher_ids == pitcher)[0][0]]}:\n")
    sequences = model_training_data[model_training_data['pitcher_id'] == pitcher]
    m = np.max(sequences['sequence_number'].values) + 1


    ### Record relevant information and pre-process data:

    pitches = sequences.iloc[:,9:16].values
    pitch_totals = np.sum(pitches, axis = 0)
    drop_axes = np.where(pitch_totals == 0)[0]
    sample_size[np.where(pitcher_ids == pitcher)] = pitches.shape[0]
    num_pitches = len(np.delete(pitch_totals, drop_axes))
    arsenal_size[np.where(pitcher_ids == pitcher)] = num_pitches
    dropped_cols = []

    for c in range(1,8):
        if np.sum(sequences[f'p_{c}'].values) == 0:
            dropped_cols.append(f'p_p{c}')
            dropped_cols.append(f'b_p{c}')
            dropped_cols.append(f'p_{c}')
    sequences = sequences.drop(columns = dropped_cols)


    ### Build data arrays for all pitch sequences:

    first_sequence = sequences[sequences['sequence_number'] == 1].iloc[:, 9:].values
    x_values, context_values, y_values = sequence_builder(first_sequence, num_pitches)

    for s in range(2,m):
        seq = sequences[sequences['sequence_number'] == s].iloc[:, 9:].values
        x, c, y = sequence_builder(seq, num_pitches)

        x_values = np.concatenate((x_values, x))
        context_values = np.concatenate((context_values, c))
        y_values = np.concatenate((y_values, y))

    train_x, test_x, train_c, test_c, train_y, test_y = train_test_split(x_values, context_values, y_values, random_state = 12345)


    ### Build RNN model:

    # Model inputs
    pitch_sequence = keras.Input(shape = (train_x.shape[1], train_x.shape[2]))                # Sequence inputs
    pitch_context = keras.Input(shape = (train_c.shape[1],))                                  # Context inputs
    
    # RNN Layers
    lstm_1 = layers.LSTM(128, return_sequences = True, dropout = 0.1)(pitch_sequence)         # LSTM layers
    lstm_2 = layers.LSTM(128, return_sequences = True, dropout = 0.1)(lstm_1)
    lstm_3 = layers.LSTM(128, return_sequences = False, dropout = 0.1)(lstm_2)

    combined_features = layers.Concatenate()([lstm_3, pitch_context])                         # Combine output with context data

    # Dense layers
    dense_1 = layers.Dense(512, activation = 'relu')(combined_features)                       # Dense layers with ReLU activation
    bn_1 = layers.BatchNormalization()(dense_1)
    drop_1 = layers.Dropout(rate = 0.1)(bn_1)
    dense_2 = layers.Dense(256, activation = 'relu')(drop_1)
    bn_2 = layers.BatchNormalization()(dense_2)
    drop_2 = layers.Dropout(rate = 0.1)(bn_2)
    dense_3 = layers.Dense(128, activation = 'relu')(drop_2)
    output = layers.Dense(num_pitches, activation = 'softmax')(dense_3)                       # Output layer with softmax activation

    
    ### Fit RNN model:

    model = keras.Model(inputs = [pitch_sequence, pitch_context], outputs = output)
    model.compile(optimizer = optimizers.Adam(learning_rate = 0.0001), loss = CategoricalCrossentropy(), metrics = ['accuracy'])
    history = model.fit([train_x, train_c], train_y, epochs = 20, batch_size = 32, validation_data = ([test_x, test_c], test_y))


    ### Record training and testing accuracies:

    train_accuracy = np.round(history.history['accuracy'][np.argmax(history.history['val_accuracy'])], 4)
    test_accuracy = np.round(history.history['val_accuracy'][np.argmax(history.history['val_accuracy'])], 4)
    best_epoch = np.argmax(history.history['val_accuracy'])


    ### Report results of training:

    training_accuracies[np.where(pitcher_ids == pitcher)] = train_accuracy
    testing_accuracies[np.where(pitcher_ids == pitcher)] = test_accuracy
    best_epochs [np.where(pitcher_ids == pitcher)] = best_epoch


### Report the results of the models:

print("\n--------------------\n\n  Training Results\n\n--------------------\n")
print(f"Lowest Training Accuracy:  {pitcher_names[np.argmin(training_accuracies)]} - {np.round(training_accuracies[np.argmin(training_accuracies)] * 100, 2)}%")
print(f"Highest Training Accuracy: {pitcher_names[np.argmax(training_accuracies)]} - {np.round(training_accuracies[np.argmax(training_accuracies)] * 100, 2)}%")
print(f"Mean Training Accuracy:    {np.round(np.mean(training_accuracies) * 100, 2)}%")
print(f"Training Accuracy SD:      {np.round(np.std(training_accuracies) * 100, 2)}\n")

print(f"Lowest Testing Accuracy:   {pitcher_names[np.argmin(testing_accuracies)]} - {np.round(testing_accuracies[np.argmin(testing_accuracies)] * 100, 2)}%")
print(f"Highest Testing Accuracy:  {pitcher_names[np.argmax(testing_accuracies)]} - {np.round(testing_accuracies[np.argmax(testing_accuracies)] * 100, 2)}%")
print(f"Mean Testing Accuracy:     {np.round(np.mean(testing_accuracies) * 100, 2)}%")
print(f"Testing Accuracy SD:       {np.round(np.std(testing_accuracies * 100), 2)}")
print(f"Mean Best Epoch:           {np.round(np.mean(best_epochs))}\n\n--------------------\n")