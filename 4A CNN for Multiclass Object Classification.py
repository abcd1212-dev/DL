# 1. IMPORT LIBRARIES

# TensorFlow → backend engine for deep learning
# Keras → high-level API to build neural networks easily
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# NumPy → used for numerical operations
import numpy as np

# Matplotlib → used for visualization (graphs + images)
import matplotlib.pyplot as plt

# 2. LOAD DATASET

# Fashion-MNIST dataset → contains 28x28 grayscale images of clothing
# X_train, X_test → image data
# y_train, y_test → labels (0–9 categories)
(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()


# 3. VISUALIZE SAMPLE IMAGE

# Display first image from dataset
plt.imshow(X_train[0], cmap='gray')   # show image in grayscale
plt.title(f"Label: {y_train[0]}")     # show corresponding label
plt.axis('off')                       # remove axis
plt.show()


# 4. DATA PREPROCESSING

# Normalize pixel values
# Convert range from [0,255] → [0,1] for better training stability
X_train = X_train / 255.0
X_test  = X_test / 255.0

# Reshape data for CNN
# CNN expects 4D input → (samples, height, width, channels)
# Since grayscale images → channels = 1
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)


# 5. BUILD CNN MODEL

# Sequential model → layers are added one after another
model = models.Sequential([

    # First Convolution Layer
    # 32 filters of size 3x3 → extracts basic features like edges
    # ReLU activation → introduces non-linearity
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),

    # MaxPooling Layer
    # Reduces spatial size → (28x28 → 14x14 approx)
    # Keeps important features and reduces computation
    layers.MaxPooling2D((2,2)),

    # Second Convolution Layer
    # 64 filters → captures more complex patterns
    layers.Conv2D(64, (3,3), activation='relu'),

    # Second MaxPooling
    layers.MaxPooling2D((2,2)),

    # Flatten Layer
    # Converts 2D feature maps into 1D vector
    layers.Flatten(),

    # Dropout Layer
    # Randomly disables 50% neurons during training
    # Prevents overfitting
    layers.Dropout(0.5),

    # Dense Layer
    # Fully connected layer → learns complex relationships
    layers.Dense(64, activation='relu'),

    # Output Layer
    # 10 neurons → 10 classes
    # Softmax → gives probability distribution
    layers.Dense(10, activation='softmax')
])


# 6. MODEL SUMMARY

# Displays structure, output shapes, and parameters
model.summary()


# 7. COMPILE MODEL

# optimizer='adam' → efficient weight update algorithm
# loss='sparse_categorical_crossentropy' → for integer labels
# metrics=['accuracy'] → evaluates performance
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 8. TRAIN MODEL

# Train the model for 20 epochs (UPDATED 🔥)
# validation_data → checks performance on test data during training
history = model.fit(X_train, y_train,
                    epochs=20,
                    validation_data=(X_test, y_test))


# 9. PLOT TRAINING GRAPHS

# Plot Accuracy Graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# Plot Loss Graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()


# 10. EVALUATE MODEL

# Evaluate performance on test dataset
test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_acc)


# 11. PREDICTIONS VISUALIZATION

# Predict classes for test data
predictions = model.predict(X_test)

# Display first 9 predictions
plt.figure(figsize=(8,8))

for i in range(9):
    plt.subplot(3,3,i+1)

    # Show image
    plt.imshow(X_test[i].reshape(28,28), cmap='gray')

    # Get predicted label
    pred_label = np.argmax(predictions[i])

    # True label
    true_label = y_test[i]

    # Green = correct, Red = wrong
    color = 'green' if pred_label == true_label else 'red'

    plt.title(f"P:{pred_label} / T:{true_label}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()
