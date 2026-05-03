import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load MNIST dataset
# - X_train: 60,000 training images
# - X_test: 10,000 testing images
# - Each image is 28x28 pixels (grayscale)
# - y_train, y_test: labels (digits 0–9)
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# Normalize pixel values
# - Original range: 0 to 255
# - New range: 0 to 1 → helps model train faster and more stably
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape input for CNN
# - CNN expects 4D input: (samples, height, width, channels)
# - MNIST is grayscale → channels = 1
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Create CNN model using Sequential API
# - Layers are added one after another
model = models.Sequential([

    # Convolution Layer 1
    # - 32 filters scan the image using 3x3 kernels
    # - Learns simple patterns like edges and lines
    # - ReLU activation introduces non-linearity
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),

    # MaxPooling Layer
    # - Reduces spatial size (downsampling)
    # - Keeps strongest features, reduces computation
    layers.MaxPooling2D((2,2)),

    # Dropout Layer
    # - Randomly disables 25% neurons during training
    # - Prevents overfitting (memorization)
    layers.Dropout(0.25),

    # Convolution Layer 2
    # - 64 filters → learns more complex patterns (curves, shapes)
    layers.Conv2D(64, (3,3), activation='relu'),

    # MaxPooling again to further reduce size
    layers.MaxPooling2D((2,2)),

    # Flatten Layer
    # - Converts 2D feature maps into 1D vector
    # - Required before Dense (fully connected) layers
    layers.Flatten(),

    # Dense Layer
    # - 128 neurons → learns high-level relationships
    layers.Dense(128, activation='relu'),

    # Dropout Layer
    # - Drops 50% neurons → strong regularization
    layers.Dropout(0.5),

    # Output Layer
    # - 10 neurons (one for each digit: 0–9)
    # - Softmax outputs probability distribution
    layers.Dense(10, activation='softmax')
])

# Compile the model
# - optimizer='adam': adjusts learning rate automatically
# - loss='sparse_categorical_crossentropy':
#     used when labels are integers (0–9, not one-hot encoded)
# - metrics='accuracy': tracks prediction correctness
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
# - epochs=5: dataset passed 5 times through the model
# - validation_data: evaluates performance on unseen test data
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate model performance on test data
# - Returns loss and accuracy
loss, acc = model.evaluate(X_test, y_test)

# Print final accuracy
print("Test Accuracy:", acc)
