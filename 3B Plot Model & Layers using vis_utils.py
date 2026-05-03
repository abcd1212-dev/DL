
# STEP 1: Import required libraries
from tensorflow.keras.utils import plot_model   # used to visualize model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# STEP 2: Create the Neural Network Model
model = Sequential()

# Input Layer
# shape=(2,) means model expects 2 input features (like XOR input)
model.add(Input(shape=(2,)))

# Hidden Layer 1
# Dense = fully connected layer
# units=4 → 4 neurons
# activation='relu' → introduces non-linearity
model.add(Dense(4, activation='relu'))

# Hidden Layer 2
model.add(Dense(4, activation='relu'))

# Output Layer
# units=1 → single output (binary classification)
# activation='sigmoid' → output between 0 and 1
model.add(Dense(1, activation='sigmoid'))

# plot_model() creates a diagram of the neural network
plot_model(
    model,
    to_file='model.png',        # file name where diagram will be saved
    show_shapes=True,           # shows input/output dimensions of each layer
    show_layer_names=True    )   # shows layer names in diagram

# STEP 4: Confirmation message
print("Model diagram saved as 'model.png'")
