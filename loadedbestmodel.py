
import tensorflow as tf
from keras import layers, models
from sklearn.metrics import accuracy_score

# Here i will work in the mnist_test.csv after saving the best model so i will load the best model first ,then i will preprocess the mnist_test.csv then i will Use the loaded model to make predictions on the preprocessed test data.
import pandas as pd
import numpy as np
# Load the saved model
loaded_model = tf.keras.models.load_model('/content/drive/MyDrive/Dataset/1.best_model')

# Read the test data of mnist
test_data = pd.read_csv('/content/drive/MyDrive/Dataset/mnist_test.csv')

# Separate labels (digits) from pixel values
test_labels = test_data['label']
test_pixels = test_data.drop('label', axis=1)

# Normalize pixel values by dividing by 255
test_pixels_normalized = test_pixels / 255.0

# Reshape images to 28x28 dimensions
X_test_plot = test_pixels_normalized.values.reshape(-1, 28, 28)

# Import any additional libraries needed for evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix
# Evaluate the model on the testing set
test_pred_dropout = loaded_model.predict(X_test_plot).argmax(axis=1)

# Calculate accuracy on the testing set
accuracy_test_dropout = accuracy_score(test_labels, test_pred_dropout)

# Display accuracy
print(f"Accuracy on Testing Set: {accuracy_test_dropout}")

# Create a confusion matrix
conf_matrix = confusion_matrix(test_labels, test_pred_dropout)

# Display the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)