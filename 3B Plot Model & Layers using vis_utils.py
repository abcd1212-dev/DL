# Import Sequential model from Keras
# Sequential model is used when layers are arranged one after another
from tensorflow.keras.models import Sequential

# Import required layers
# Dense = Fully connected layer
# Flatten = Converts 2D data into 1D
# Input = Defines input shape
from tensorflow.keras.layers import Dense, Flatten, Input

# Import Fashion-MNIST dataset
# Contains grayscale images of clothing items (10 classes)
from tensorflow.keras.datasets import fashion_mnist

# Import matplotlib for graphs
import matplotlib.pyplot as plt

# Import plot_model for visualization
from tensorflow.keras.utils import plot_model


# Load dataset
# X_train = training images
# y_train = training labels
# X_test = testing images
# y_test = testing labels
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# Normalize pixel values from 0–255 to 0–1
X_train = X_train / 255.0
X_test = X_test / 255.0


# Create Sequential model
model = Sequential()


# Input layer (28x28 image)
model.add(Input(shape=(28,28)))


# Flatten layer (convert 2D → 1D)
model.add(Flatten())


# First hidden layer (128 neurons, ReLU activation)
model.add(Dense(128, activation='relu'))


# Second hidden layer (64 neurons, ReLU activation)
model.add(Dense(64, activation='relu'))


# Output layer (10 classes, Softmax activation)
model.add(Dense(10, activation='softmax'))


# Compile model
# optimizer → updates weights
# loss → calculates error
# accuracy → performance metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Display model summary (text format)
model.summary()


# Train model
# epochs = number of times data passes through model
# validation_split = 20% data used for validation
history = model.fit(X_train, y_train,
                    epochs=10,
                    validation_split=0.2)


# Evaluate model on test data
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)


# Plot model architecture
# show_shapes=True shows input and output dimensions
# show_layer_names=True shows names of layers
plot_model(model,
           show_shapes=True,
           show_layer_names=True)


# Plot Accuracy Graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch Number")
plt.ylabel("Accuracy")
plt.legend(["Training", "Validation"])

plt.show()

from tensorflow.keras.utils import plot_model
     
plot_model(model, show_shapes=True, show_layer_names=True)
