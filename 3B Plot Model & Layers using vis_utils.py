# Import Sequential model
from tensorflow.keras.models import Sequential

# Import required layers
from tensorflow.keras.layers import Dense, Flatten, Input

# Import plot_model function
# Used to display model diagram visually
from tensorflow.keras.utils import plot_model


# Create Sequential model
model = Sequential()


# Input layer
# Input image size = 28 x 28
model.add(Input(shape=(28,28)))


# Flatten layer
# Converts 2D image into 1D vector
model.add(Flatten())


# First hidden layer
# 128 neurons with ReLU activation
model.add(Dense(128, activation='relu'))


# Second hidden layer
# 64 neurons with ReLU activation
model.add(Dense(64, activation='relu'))


# Output layer
# 10 neurons for 10 classes
# Softmax gives class probabilities
model.add(Dense(10, activation='softmax'))


# Display textual summary of model
# Includes layer names, output shapes, total parameters
model.summary()


# Plot graphical structure of model
# show_shapes=True shows input and output dimensions
# show_layer_names=True shows names of layers
plot_model(model,
           show_shapes=True,
           show_layer_names=True)
