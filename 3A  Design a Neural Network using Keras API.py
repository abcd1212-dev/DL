# Import Sequential model from Keras
# Sequential model is used when layers are arranged one after another
from tensorflow.keras.models import Sequential

# Import required layers
# Dense = Fully connected layer
# Flatten = Converts 2D data into 1D
# Input = Defines input shape
from tensorflow.keras.layers import Dense, Flatten, Input

# Import Fashion-MNIST dataset
# It is an inbuilt dataset available in Keras
# Contains images of clothes like shirts, shoes, bags etc.
from tensorflow.keras.datasets import fashion_mnist

# Import matplotlib for plotting graphs
import matplotlib.pyplot as plt


# Load dataset
# X_train = training images
# y_train = training labels
# X_test = testing images
# y_test = testing labels
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# Dataset images contain pixel values from 0 to 255
# Normalize values into range 0 to 1
# This improves speed and accuracy of training
X_train = X_train / 255.0
X_test = X_test / 255.0


# Create empty Sequential model
model = Sequential()


# Input layer
# Each image size is 28 rows × 28 columns
model.add(Input(shape=(28,28)))


# Flatten layer
# Converts 28x28 matrix into single vector of 784 values
# Neural network Dense layer needs 1D input
model.add(Flatten())


# First Hidden Layer
# 128 neurons are used
# ReLU activation removes negative values
# Helps model learn complex patterns
model.add(Dense(128, activation='relu'))


# Second Hidden Layer
# 64 neurons used
# Learns deeper features from previous layer
model.add(Dense(64, activation='relu'))


# Output Layer
# 10 neurons because dataset has 10 classes
# Softmax converts output into probabilities
# Highest probability becomes final prediction
model.add(Dense(10, activation='softmax'))


# Compile model
# optimizer='adam' updates weights efficiently
# loss function checks prediction error
# accuracy used to measure correct predictions
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Display complete model structure
# Shows layer names, shapes and parameters
model.summary()


# Train the model
# epochs=20 means complete dataset passes through model 20 times
# validation_split=0.1 means 10% training data used for validation
history = model.fit(X_train, y_train,
                    epochs=20,
                    validation_split=0.1)


# Evaluate model using test dataset
# Gives final loss and accuracy
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)


# Plot graph of training accuracy and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch Number")
plt.ylabel("Accuracy")
plt.legend(["Training", "Validation"])

plt.show()
