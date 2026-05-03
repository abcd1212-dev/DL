# Import TensorFlow and Keras modules
# TensorFlow = backend engine, Keras = high-level API to build neural networks
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load dataset
# Fashion-MNIST → 28x28 grayscale images of clothing (10 categories)
# X = images (input), y = labels (class index 0–9)
(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()

# Normalize pixel values
# Convert from range [0,255] → [0,1]
# Helps faster convergence and stable gradient updates
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape data for CNN
# Current shape: (samples, 28, 28)
# CNN expects: (samples, height, width, channels)
# Since grayscale → channels = 1
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

# Create Sequential model → layers are added one after another
model = models.Sequential([

    # First Convolution Layer
    layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),

    # MaxPooling Layer
    # Reduces spatial size (28x28 → ~14x14)
    # Keeps strongest features, reduces computation and overfitting
    layers.MaxPooling2D(2),

    # Second Convolution Layer
    # 64 filters → learn more complex patterns (shapes, textures)
    layers.Conv2D(64, 3, activation='relu'),

    # Further downsampling (reduces size again)
    layers.MaxPooling2D(2),

    # Flatten Layer
    # Converts 2D feature maps into 1D vector
    # Required before feeding into Dense (fully connected) layers
    layers.Flatten(),

    # Dense Layer
    # Learns relationships between extracted features
    # Acts as decision-making part of the network
    layers.Dense(64, activation='relu'),

    # Output Layer
    # 10 neurons → one for each class
    # Softmax → converts output into probabilities (sum = 1)
    layers.Dense(10, activation='softmax')
])

# Compile model
# optimizer='adam' → adjusts weights efficiently during training
# loss='sparse_categorical_crossentropy' → correct for integer labels
# metrics=['accuracy'] → measure performance
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
# epochs=5 → model sees entire dataset 5 times
# validation_data → evaluates performance on unseen test data during training
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate model on test data
# Returns loss and accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)

# Print final accuracy
print("Test Accuracy:", test_acc)
