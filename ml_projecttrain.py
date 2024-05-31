
import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt

# Load the training data
train_data = pd.read_csv('/content/drive/MyDrive/Dataset/mnist_train.csv')

# Separate labels (digits) from pixel values
labels = train_data['label']
pixels = train_data.drop('label', axis=1)

print(train_data.head())
print(train_data.info())

# Normalize pixel values by dividing by 255
pixels_normalized = pixels / 255.0

# Reshape images to 28x28 dimensions
X_train_plot = pixels_normalized.values.reshape(-1, 28, 28)

# Check unique classes and number of features
unique_classes = labels.unique()
num_features = pixels_normalized.shape[1]
print("Unique Classes:", unique_classes)
print("Number of Features:", num_features)

# Check for missing values
missing_values = pixels.isnull().sum().sum()
if missing_values == 0:
    print("No missing values in the dataset")
else:
    print("There are missing values in the dataset")

# Visualize more images to verify reshaping correctness
fig = plt.figure(figsize=(8, 8))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_train_plot[i], cmap='gray')
    plt.title(f"Digit {labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Function return digit in grayscale
def plot_digit(digit, dem = 28, font_size = 12):
    max_ax = font_size * dem

    fig = plt.figure(figsize=(13, 13))
    plt.xlim([0, max_ax])
    plt.ylim([0, max_ax])
    plt.axis('off')
    black = '#000000'

    for idx in range(dem):
        for jdx in range(dem):

            t = plt.text(idx * font_size, max_ax - jdx*font_size, digit[jdx][idx], fontsize = font_size, color = black)
            c = digit[jdx][idx] / 255.
            t.set_bbox(dict(facecolor=(c, c, c), alpha = 0.5, edgecolor = 'black'))

    plt.show()
rand_number = random.randint(0, len(labels))
print(labels[rand_number])
plot_digit(X_train_plot[rand_number])

from sklearn.model_selection import train_test_split

# Split the data into training and validation sets (80% train, 20% validation)
train_images, val_images, train_labels, val_labels = train_test_split(
    X_train_plot, labels, test_size=0.2, random_state=42
)

# Check the shapes of the split datasets
print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)
print("Validation images shape:", val_images.shape)
print("Validation labels shape:", val_labels.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Initialize K-NN classifier
knn = KNeighborsClassifier()

# Define hyperparameters to search
param_grid = {'n_neighbors': [3, 5, 7, 9]}  # Example: Searching for the best number of neighbors

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(train_images.reshape(-1, 28*28), train_labels)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

from sklearn.metrics import confusion_matrix

# Evaluate the model
knn_best = grid_search.best_estimator_
knn_best.fit(train_images.reshape(-1, 28*28), train_labels)

train_accuracy = knn_best.score(train_images.reshape(-1, 28*28), train_labels)
val_accuracy = knn_best.score(val_images.reshape(-1, 28*28), val_labels)

print(f"Train Accuracy (K-NN): {train_accuracy:.4f}")
print(f"Validation Accuracy (K-NN): {val_accuracy:.4f}")

# Get confusion matrix
val_pred = knn_best.predict(val_images.reshape(-1, 28*28))
conf_matrix = confusion_matrix(val_labels, val_pred)
print("Confusion Matrix (K-NN):")
print(conf_matrix)

import tensorflow as tf
from keras import layers, models
from sklearn.metrics import accuracy_score

# Function to build and train the first arch for  ANN with dropout
def train_ann_with_dropout(architecture, train_images, train_labels, val_images, val_labels, epochs=10, batch_size=32, learning_rate=0.001, dropout_rate=0.2):
    # Build the model
    model = tf.keras.models.Sequential()

    # Add Flatten layer to flatten the input images
    model.add(tf.keras.layers.Flatten())

    for layer_size in architecture:
        model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))  # Add dropout after each hidden layer

    model.add(tf.keras.layers.Dense(10, activation='softmax'))  # 10 output classes for digits 0-9

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels))

    return model, history

# Define architecture with dropout to experiment with
architecture1_dropout = [128, 64]

# Train and evaluate the architecture with dropout
model1_dropout, history1_dropout = train_ann_with_dropout(architecture1_dropout, train_images, train_labels, val_images, val_labels, batch_size=64, learning_rate=0.001, dropout_rate=0.2)

# Evaluate the model with dropout on the validation set
val_pred1_dropout = model1_dropout.predict(val_images).argmax(axis=1)
accuracy1_dropout = accuracy_score(val_labels, val_pred1_dropout)

print(f"Accuracy for Architecture 1 with Dropout: {accuracy1_dropout}")

# Function to build and train the second arc for ANN with dropout(differnt numer of hidden layers, batch size, learning rate)
def train_ann_with_dropout(architecture, train_images, train_labels, val_images, val_labels, epochs=10, batch_size=64, learning_rate=0.003, dropout_rate=0.2):
    # Build the model
    model = tf.keras.models.Sequential()

    # Add Flatten layer to flatten the input images
    model.add(tf.keras.layers.Flatten())

    for layer_size in architecture:
        model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))  # Add dropout after each hidden layer

    model.add(tf.keras.layers.Dense(10, activation='softmax'))  # 10 output classes for digits 0-9

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels))

    return model, history

# Define architecture with dropout to experiment with
architecture2_dropout = [256, 128, 64]

# Train and evaluate the architecture with dropout
model2_dropout, history2_dropout = train_ann_with_dropout(architecture2_dropout, train_images, train_labels, val_images, val_labels, batch_size=128, learning_rate=0.001, dropout_rate=0.2)

# Evaluate the model with dropout on the validation set
val_pred2_dropout = model2_dropout.predict(val_images).argmax(axis=1)
accuracy2_dropout = accuracy_score(val_labels, val_pred2_dropout)

print(f"Accuracy for Architecture 2 with Dropout: {accuracy2_dropout}")

#Confusion Matrix for the best model (model2 for ann (second architecture))
from sklearn.metrics import confusion_matrix

# Get predictions on the validation set
val_predictions = model2_dropout.predict(val_images).argmax(axis=1)

# Create the confusion matrix
conf_matrix = confusion_matrix(val_labels, val_predictions)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

model2_dropout.save('/content/drive/MyDrive/Dataset/1.best_model')