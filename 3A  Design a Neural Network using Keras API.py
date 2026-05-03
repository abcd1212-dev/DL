# Import Sequential model
# Sequential means layers are added one after another
from tensorflow.keras.models import Sequential

# Import required layers
from tensorflow.keras.layers import Dense, Flatten, Input

# Import inbuilt Fashion-MNIST dataset
from tensorflow.keras.datasets import fashion_mnist

# Import matplotlib for graphs
import matplotlib.pyplot as plt


# Load Fashion-MNIST dataset
# 28x28 grayscale images of clothes
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# Normalize pixel values from 0-255 to 0-1
# Helps faster training
X_train = X_train / 255.0
X_test = X_test / 255.0


# Create Neural Network model
model = Sequential()

# Input layer for image size 28x28
model.add(Input(shape=(28,28)))

# Flatten layer converts image into 1D vector
model.add(Flatten())

# Hidden layer 1 with 128 neurons
model.add(Dense(128, activation='relu'))

# Hidden layer 2 with 64 neurons
model.add(Dense(64, activation='relu'))

# Output layer with 10 neurons for 10 classes
model.add(Dense(10, activation='softmax'))


# Compile model
# Adam optimizer adjusts weights
# Sparse categorical crossentropy for class labels
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Show model summary
model.summary()


# Train model for 20 epochs
history = model.fit(X_train, y_train,
                    epochs=20,
                    validation_split=0.1)


# Test model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

# Plot accuracy graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])

plt.show()
