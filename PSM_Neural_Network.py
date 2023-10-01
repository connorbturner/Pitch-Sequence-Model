### Neural Network Model for Conditional Pitch Sequencing
### Author: Connor Turner

### Import relevant modules and functions

import pandas as pd
import numpy as np
import itertools
from tensorflow import keras
from keras import layers
from tensorflow.keras.optimizers.legacy import Adam
from keras.losses import CategoricalCrossentropy
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


### Import the training, validation, and testing data

model_training_data = pd.read_csv('psm_training_data.csv')
model_validation_data = pd.read_csv('psm_validation_data.csv')
model_testing_data = pd.read_csv('psm_testing_data.csv')


### Split the data sets into parameter and target vectors

training_x = model_training_data.iloc[:, 9:-7].values
training_y = model_training_data.iloc[:, -7:].values
validation_x = model_validation_data.iloc[:, 9:-7].values
validation_y = model_validation_data.iloc[:, -7:].values
testing_x = model_testing_data.iloc[:, 9:-7].values
testing_y = model_testing_data.iloc[:, -7:].values


### Set the pitch labels and calculate the class weights

pitch_labels = ['Fastball', 'Sinker', 'Cutter', 'Slider', 'Curveball', 'Changeup', 'Splitter']
calculate_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(np.argmax(training_y, axis = 1)), y = np.argmax(training_y, axis = 1))
pitch_weights = {pitch: weight for pitch, weight in enumerate(calculate_weights)}


### Build the neural network model

model = keras.Sequential([
    layers.Input(shape = (training_x.shape[1],)),   # Input layer
    layers.Dense(1024, activation = 'relu'), 
    layers.BatchNormalization(),
    layers.Dense(1024, activation = 'relu'),        # Hidden layer with ReLU activation
    layers.BatchNormalization(),
    layers.Dense(1024, activation = 'relu'), 
    layers.BatchNormalization(),
    layers.Dense(1024, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dense(1024, activation = 'relu'), 
    layers.BatchNormalization(),
    layers.Dense(512, activation = 'relu'), 
    layers.BatchNormalization(),
    layers.Dense(512, activation = 'relu'), 
    layers.BatchNormalization(),
    layers.Dense(512, activation = 'relu'), 
    layers.BatchNormalization(),
    layers.Dense(512, activation = 'relu'), 
    layers.BatchNormalization(),
    layers.Dense(256, activation = 'relu'), 
    layers.BatchNormalization(),
    layers.Dense(256, activation = 'relu'), 
    layers.BatchNormalization(),
    layers.Dense(256, activation = 'relu'), 
    layers.BatchNormalization(),
    layers.Dense(256, activation = 'relu'), 
    layers.BatchNormalization(),
    layers.Dense(256, activation = 'relu'), 
    layers.BatchNormalization(),
    layers.Dense(7, activation = 'softmax')         # Output layer with softmax activation (7 units for 7 targets)
])


### Compile and fit the model to the data
model.compile(optimizer = Adam(), loss = CategoricalCrossentropy(), metrics = ['accuracy'])
PSM = model.fit(training_x, training_y, epochs = 15, batch_size = 256, validation_data = (validation_x, validation_y))


### Plot the results of the neural network model

def PlotConfusionMatrix(cm, classes, normalize = False, title = 'Confusion Matrix', cmap = plt.cm.Blues, showAcc = True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    
    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment = 'center',
            verticalalignment = 'center',
            color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Pitch')
    plt.xlabel('Predicted Pitch')
    
    if showAcc:
        acc = 100 * (np.trace(cm) / np.sum(cm))
        title = title + " | Acc = %.2f%%" % acc
        
    plt.title(title)

def PlotModelEval(model, history, x, y, labels):
    
    # Scores for each class (can be interpreted as probabilities since we use softmax output)
    probabilities = model.predict(x)
    # Prediction (class number) for each test image
    predicted_pitch = np.expand_dims(np.argmax(probabilities,axis=1), axis=-1)
    true_pitch = np.expand_dims(np.argmax(y,axis=1), axis=-1)

    # Calculate confusion matrix
    cm = confusion_matrix(true_pitch, predicted_pitch)
    
    # Plot training history
    plt.figure(figsize = (16,6))
    plt.subplot(2,2,1)
    plt.plot(history.history['loss'], label = 'Training')
    plt.plot(history.history['val_loss'], label = 'Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc = 'upper right')
    plt.grid(True, which = 'both')

    plt.subplot(2,2,3)
    plt.plot(100 * np.array(history.history['accuracy']), label= 'Training')
    plt.plot(100 * np.array(history.history['val_accuracy']), label = 'Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Acc [%]')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True, which = 'both')
    
    # Plot confusion matrix
    plt.subplot(2,2,(2,4))
    PlotConfusionMatrix(cm, classes = labels, title = 'Confusion Matrix - Test Data')
    plt.show()

PlotModelEval(model, PSM, testing_x, testing_y, pitch_labels)