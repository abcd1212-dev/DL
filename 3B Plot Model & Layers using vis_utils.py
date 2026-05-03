# Import required classes from Keras
from keras.models import Sequential
from keras.layers import Dense, Input

# Create the model object
model = Sequential()

# Add Input layer FIRST
# shape=(2,) → model expects 2 input features (like [x1, x2])
model.add(Input(shape=(2,)))

# Hid
# Hidden Layer 1
# Dense layer with 4 neurons
# activation='relu' → introduces non-linearity (important for learning complex patterns)
model.add(Dense(4, activation='relu'))

# Hidden Layer 2
# Another Dense layer with 4 neurons
model.add(Dense(4, activation='relu'))

# Output Layer
# Dense layer with 1 neuron (binary output)
# activation='sigmoid' → gives output between 0 and 1 (used for binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
# optimizer='adam' → adjusts weights efficiently during training
# loss='binary_crossentropy' → used for binary classification problems
# metrics=['accuracy'] → used to measure model performance
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=True)
