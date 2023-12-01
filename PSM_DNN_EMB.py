### DNN Model for Conditional Pitch Sequencing; Embedded Context
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


### Import the data:

model_training_data = pd.read_csv('dnn_emb_model_data.csv')
model_training_data = model_training_data[model_training_data['game_year'] == 2021]


### Initialize vectors to record model performance:

pitcher_ids = np.unique(model_training_data['pitcher_id'])[:30]
pitcher_names = model_training_data[model_training_data['pitcher_id'].isin(pitcher_ids)]['pitcher_name'].unique()
sample_size = np.zeros((len(pitcher_ids),))
training_accuracies = np.zeros((len(pitcher_ids),))
testing_accuracies = np.zeros((len(pitcher_ids),))
improvements = np.zeros((len(pitcher_ids),))
best_epochs = np.zeros((len(pitcher_ids),))
arsenal_size = np.zeros((len(pitcher_ids),))
naive_ests = np.zeros((len(pitcher_ids),))
overfits = np.zeros((len(pitcher_ids),))


### Train model for each pitcher:

for idx, pitcher in enumerate(pitcher_ids):
    
    ### Record relevant information and pre-process data:

    print(f"\n{pitcher_names[np.where(pitcher_ids == pitcher)[0][0]]}:\n")
    data = model_training_data[model_training_data['pitcher_id'] == pitcher]
    pitches = data.iloc[:,-7:].values
    pitch_totals = np.sum(pitches, axis = 0)
    drop_axes = np.where(pitch_totals == 0)[0]
    sample_size[idx] = pitches.shape[0]
    naive_ests[idx] = np.round(np.max(pitch_totals) / sample_size[idx], 4)
    num_pitches = len(np.delete(pitch_totals, drop_axes))
    arsenal_size[np.where(pitcher_ids == pitcher)] = num_pitches
    dropped_cols = []

    for c in range(1,8):
        if np.sum(data[f'p_{c}'].values) == 0:
            dropped_cols.append(f'p_p{c}')
            dropped_cols.append(f'b_p{c}')
            dropped_cols.append(f'p_{c}')
    filtered_data = data.drop(columns = dropped_cols)
    x_values = filtered_data.iloc[:,8:-num_pitches].values
    y_values = filtered_data.iloc[:,-num_pitches:].values

    train_x, test_x, train_y, test_y = train_test_split(x_values, y_values, random_state = 12345)


    ### Build DNN model:

    # Model inputs
    pitch_features = layers.Input(shape = (train_x.shape[1],))

    # Feature embedding
    input_dims = (8, 15, 8, 15, 8, 15, 9, 13, 10, 3, 9, 4)                                                         # Input categories
    embedding_dim = 32                                                                                             # Embedding dimensions
    embedded_context = layers.Embedding(input_dims[0], embedding_dim)(pitch_features[:, (2 * num_pitches)])  

    for i in range(1, len(input_dims)):                                                                            # For each feature:
        new_embedding = layers.Embedding(input_dims[i], embedding_dim)(pitch_features[:, (2 * num_pitches) + i])   # Build embedding layer
        embedded_context += new_embedding                                                                          # Add to combined embedding layer

    embedded_pitches = layers.Concatenate()([pitch_features[:,:(2 * num_pitches)], embedded_context])              # Attach pitch probability data

    # Dense neural network layers
    dense_1 = layers.Dense(256, activation = 'relu')(embedded_pitches)                                             # Dense layer with ReLU activation
    bn_1 = layers.BatchNormalization()(dense_1)                                                                    # Batch normalization layer
    drop_1 = layers.Dropout(rate = 0.2)(bn_1)                                                                      # Dropout layer
    dense_2 = layers.Dense(256, activation = 'relu')(drop_1)
    bn_2 = layers.BatchNormalization()(dense_2)
    drop_2 = layers.Dropout(rate = 0.2)(bn_2)
    dense_3 = layers.Dense(256, activation = 'relu')(drop_2)  
    bn_3 = layers.BatchNormalization()(dense_3)
    drop_3 = layers.Dropout(rate = 0.2)(bn_3)
    dense_4 = layers.Dense(128, activation = 'relu')(drop_3)  
    bn_4 = layers.BatchNormalization()(dense_4)
    drop_4 = layers.Dropout(rate = 0.2)(bn_4)
    dense_5 = layers.Dense(64, activation = 'relu')(drop_4)
    output = layers.Dense(num_pitches, activation = 'softmax')(dense_5)                                            # Output layer with softmax activation


    ### Fit DNN Model:

    model = keras.Model(inputs = pitch_features, outputs = output)
    model.compile(optimizer = optimizers.Adam(learning_rate = 0.0001), loss = CategoricalCrossentropy(), metrics = ['accuracy'])
    history = model.fit(train_x, train_y, epochs = 20, batch_size = 32, validation_data = (test_x, test_y))


    ### Record training and testing accuracies:

    train_accuracy = np.round(history.history['accuracy'][np.argmax(history.history['val_accuracy'])], 4)
    test_accuracy = np.round(history.history['val_accuracy'][np.argmax(history.history['val_accuracy'])], 4)
    improvement = np.round(test_accuracy - naive_ests[idx], 4)
    overfit = train_accuracy - test_accuracy
    best_epoch = np.argmax(history.history['val_accuracy'])


    ### Report results of training:

    training_accuracies[np.where(pitcher_ids == pitcher)] = train_accuracy
    testing_accuracies[np.where(pitcher_ids == pitcher)] = test_accuracy
    improvements[np.where(pitcher_ids == pitcher)] = improvement
    overfits[np.where(pitcher_ids == pitcher)] = overfit
    best_epochs [np.where(pitcher_ids == pitcher)] = best_epoch


### Report the results of the models

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


### Save results to data frame and export:

result_data = {'pitcher_name':pitcher_names, 'pitcher_id':pitcher_ids, 'sample_size':sample_size, 
               'num_pitch_types':arsenal_size, 'train_acc':training_accuracies, 'test_acc':testing_accuracies, 
               'best_epoch':best_epochs, 'naive_est':naive_ests, 'improvement':improvements, 'overfit':overfits}

results = pd.DataFrame(result_data)
results.to_csv(path_or_buf = 'DNN_EMB_results.csv')