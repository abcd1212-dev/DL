# IMPORT LIBRARIES

# TensorFlow provides core deep learning operations
# Keras (inside TensorFlow) provides easy-to-use APIs for building models
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# NumPy is used for numerical operations and array handling
import numpy as np

# Matplotlib is used for plotting images and graphs
import matplotlib.pyplot as plt


# LOAD MNIST DATASET

# MNIST contains grayscale images of handwritten digits (0–9)
# X_train: 60,000 training images
# y_train: labels corresponding to training images
# X_test: 10,000 testing images
# y_test: labels corresponding to test images
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()


# DISPLAY SAMPLE IMAGE

# Show the first image from the training dataset
# cmap='gray' ensures the image is displayed in grayscale
plt.imshow(X_train[0], cmap='gray')

# Display the label of the image as the title
plt.title(f"Label: {y_train[0]}")

# Hide axis values for cleaner visualization
plt.axis('off')

# Render the image
plt.show()


# DATA PREPROCESSING

# Normalize pixel values from range [0,255] to [0,1]
# This improves training stability and convergence speed
X_train = X_train / 255.0
X_test  = X_test / 255.0

# Reshape images to add channel dimension
# CNN expects input in 4D format: (samples, height, width, channels)
# Since images are grayscale, channel size = 1
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)


# BUILD CNN MODEL

# Sequential model stacks layers one after another
model = models.Sequential([

    # First convolutional layer
    # Applies 32 filters of size 3x3 to detect basic features like edges
    # ReLU activation introduces non-linearity and helps learning complex patterns
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),

    # MaxPooling layer reduces spatial dimensions (downsampling)
    # Keeps important features and reduces computation
    layers.MaxPooling2D((2,2)),

    # Dropout randomly disables 25% of neurons during training
    # This helps prevent overfitting
    layers.Dropout(0.25),

    # Second convolutional layer
    # Uses 64 filters to learn more complex features like shapes and textures
    layers.Conv2D(64, (3,3), activation='relu'),

    # Second pooling layer further reduces size of feature maps
    layers.MaxPooling2D((2,2)),

    # Flatten layer converts 2D feature maps into a 1D vector
    # This is required before feeding into fully connected layers
    layers.Flatten(),

    # Dense (fully connected) layer
    # Learns high-level relationships between extracted features
    layers.Dense(128, activation='relu'),

    # Dropout randomly disables 50% neurons to improve generalization
    layers.Dropout(0.5),

    # Output layer
    # 10 neurons correspond to 10 digit classes (0–9)
    # Softmax activation converts outputs into probabilities
    layers.Dense(10, activation='softmax')
])


# MODEL SUMMARY

# Displays model architecture, layer output shapes, and number of parameters
model.summary()


# COMPILE MODEL

# Adam optimizer adjusts learning rate during training
# Sparse categorical crossentropy is used because labels are integers (0–9)
# Accuracy is used as the evaluation metric
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# TRAIN MODEL

# Train the model for 20 epochs
# Each epoch means the model sees the entire dataset once
# validation_data evaluates performance on unseen test data during training
history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_test, y_test)
)


# PLOT TRAINING PERFORMANCE

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Graph')
plt.show()

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Graph')
plt.show()


# MODEL EVALUATION

# Evaluate model performance on test dataset
# Returns loss value and accuracy
loss, acc = model.evaluate(X_test, y_test)

# Print final accuracy
print("Test Accuracy:", acc)


# PREDICTIONS VISUALIZATION

# Generate predictions for test dataset
predictions = model.predict(X_test)

# Create a grid to display predictions
plt.figure(figsize=(8,8))

for i in range(9):
    plt.subplot(3,3,i+1)

    # Display the test image
    plt.imshow(X_test[i].reshape(28,28), cmap='gray')

    # Predicted label is the index with highest probability
    pred_label = np.argmax(predictions[i])

    # True label from dataset
    true_label = y_test[i]

    # Use green color if prediction is correct, otherwise red
    color = 'green' if pred_label == true_label else 'red'

    # Show predicted and actual labels
    plt.title(f"P:{pred_label} / T:{true_label}", color=color)

    # Hide axis
    plt.axis('off')

# Adjust layout for better spacing
plt.tight_layout()

# Display prediction results
plt.show()
