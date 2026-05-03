# IMPORT LIBRARIES

# TensorFlow → used for building deep learning models
import tensorflow as tf

# Keras layers → used to define neural network layers
from tensorflow.keras import layers

# NumPy → used for random number generation and array handling
import numpy as np

# Matplotlib → used to display generated images
import matplotlib.pyplot as plt


# LOAD MNIST DATASET

# MNIST contains handwritten digit images (28x28 pixels)
# Labels are ignored (_) because GAN is unsupervised
(X_train, _), _ = tf.keras.datasets.mnist.load_data()


# NORMALIZE DATA

# Convert pixel values from [0,255] → [-1,1]
# This is required because generator uses tanh activation
# Matching ranges improves training stability
X_train = (X_train - 127.5) / 127.5


# RESHAPE DATA

# Convert shape from (28,28) → (28,28,1)
# Channel dimension is required for neural networks
X_train = X_train.reshape(-1, 28, 28, 1)


# GENERATOR MODEL

# Generator takes random noise and generates fake images
def build_generator():
    model = tf.keras.Sequential([

        # Input: random noise vector of size 100
        layers.Dense(128, activation='relu', input_shape=(100,)),

        # Output layer: produces 784 values (28x28 image flattened)
        # tanh activation outputs values in [-1,1]
        layers.Dense(784, activation='tanh'),

        # Reshape flattened output into image format (28x28x1)
        layers.Reshape((28, 28, 1))
    ])
    return model


# DISCRIMINATOR MODEL

# Discriminator checks whether image is real or fake
def build_discriminator():
    model = tf.keras.Sequential([

        # Flatten image into vector for processing
        layers.Flatten(input_shape=(28, 28, 1)),

        # Dense layer learns features from image
        layers.Dense(128, activation='relu'),

        # Output layer: sigmoid gives probability (0=fake, 1=real)
        layers.Dense(1, activation='sigmoid')
    ])
    return model


# CREATE MODELS

generator = build_generator()
discriminator = build_discriminator()


# COMPILE DISCRIMINATOR

# Binary crossentropy is used for real vs fake classification
discriminator.compile(optimizer='adam', loss='binary_crossentropy')


# BUILD GAN MODEL

# Freeze discriminator during GAN training
# Only generator will be trained in GAN step
discriminator.trainable = False

# Input to GAN is random noise
gan_input = tf.keras.Input(shape=(100,))

# Generator creates fake image
fake_image = generator(gan_input)

# Discriminator evaluates fake image
gan_output = discriminator(fake_image)

# Combine into one GAN model
gan = tf.keras.Model(gan_input, gan_output)

# Compile GAN
gan.compile(optimizer='adam', loss='binary_crossentropy')


# TRAINING LOOP

epochs = 1000
batch_size = 32

for epoch in range(epochs):

    # -------------------------
    # STEP 1: TRAIN DISCRIMINATOR
    # -------------------------

    # Select random real images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]

    # Generate fake images from random noise
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_images = generator.predict(noise, verbose=0)

    # Enable discriminator training
    discriminator.trainable = True

    # Train on real images (label = 1)
    d_loss_real = discriminator.train_on_batch(
        real_images,
        np.ones((batch_size, 1))
    )

    # Train on fake images (label = 0)
    d_loss_fake = discriminator.train_on_batch(
        fake_images,
        np.zeros((batch_size, 1))
    )

    # -------------------------
    # STEP 2: TRAIN GENERATOR
    # -------------------------

    # Freeze discriminator
    discriminator.trainable = False

    # Generate new noise
    noise = np.random.normal(0, 1, (batch_size, 100))

    # Generator tries to fool discriminator → label = 1
    g_loss = gan.train_on_batch(
        noise,
        np.ones((batch_size, 1))
    )

    # Print progress every 200 epochs
    if epoch % 200 == 0:
        print("Epoch:", epoch, "D Loss:", d_loss_real, "G Loss:", g_loss)


# GENERATE SAMPLE IMAGE

# Create random noise
noise = np.random.normal(0, 1, (1, 100))

# Generate fake image
gen_img = generator.predict(noise, verbose=0)

# Display generated image
plt.imshow(gen_img[0, :, :, 0], cmap='gray')
plt.title("Generated Image")
plt.show()
