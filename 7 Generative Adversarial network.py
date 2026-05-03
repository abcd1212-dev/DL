# STEP 1: IMPORT LIBRARIES
# TensorFlow/Keras → build neural networks
# NumPy → numerical operations
# Matplotlib → visualization

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

# Set random seed so results are reproducible
tf.random.set_seed(42)
np.random.seed(42)


# STEP 2: LOAD AND PREPROCESS MNIST DATASET

# Load dataset → returns (train_images, train_labels), (test_images, test_labels)
# We only need images, so labels are ignored using "_"
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from [0,255] to [-1,1]
# GAN works better when data is centered around 0
X_train = (X_train.astype('float32') - 127.5) / 127.5

# Reshape to (28,28,1) → grayscale image format
X_train = X_train.reshape(-1, 28, 28, 1)


# STEP 3: BUILD GENERATOR
# Generator takes random noise and generates fake images

def build_generator(latent_dim):

    model = models.Sequential()

    # Input: random noise vector
    model.add(layers.Input(shape=(latent_dim,)))

    # Hidden layers → learn patterns
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))

    # Output layer → 784 values (28x28 image)
    model.add(layers.Dense(28 * 28, activation='tanh'))

    # Convert flat vector → image
    model.add(layers.Reshape((28, 28, 1)))

    return model


# STEP 4: BUILD DISCRIMINATOR
# Discriminator classifies image as REAL (1) or FAKE (0)

def build_discriminator():

    model = models.Sequential()

    # Convert image to vector
    model.add(layers.Flatten(input_shape=(28, 28, 1)))

    # Hidden layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))

    # Output → probability
    model.add(layers.Dense(1, activation='sigmoid'))

    # GAN-specific optimizer
    optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


# STEP 5: BUILD GAN (COMBINED MODEL)
# Generator + Discriminator
# Discriminator is frozen during generator training

def build_gan(generator, discriminator, latent_dim):

    discriminator.trainable = False

    gan_input = tf.keras.Input(shape=(latent_dim,))
    fake_image = generator(gan_input)
    gan_output = discriminator(fake_image)

    model = tf.keras.Model(gan_input, gan_output)

    optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model


# STEP 6: HELPER FUNCTIONS

# Get real images
def generate_real_samples(n_samples):
    idx = np.random.randint(0, X_train.shape[0], n_samples)
    X = X_train[idx]
    y = np.ones((n_samples, 1))  # label = real
    return X, y


# Generate random noise
def generate_latent_points(latent_dim, n_samples):
    return np.random.randn(n_samples, latent_dim)


# STEP 7: TRAIN GAN

def train_gan(generator, discriminator, gan_model,
              latent_dim, epochs=1000, batch_size=64):

    half_batch = batch_size // 2

    d_losses = []
    g_losses = []

    for epoch in range(epochs):

        # Train Discriminator

        X_real, y_real = generate_real_samples(half_batch)

        noise = generate_latent_points(latent_dim, half_batch)
        X_fake = generator.predict(noise, verbose=0)
        y_fake = np.zeros((half_batch, 1))

        discriminator.trainable = True

        d_loss_real, _ = discriminator.train_on_batch(X_real, y_real)
        d_loss_fake, _ = discriminator.train_on_batch(X_fake, y_fake)

        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # Train Generator

        noise = generate_latent_points(latent_dim, batch_size)
        y_gan = np.ones((batch_size, 1))  # generator wants output as real

        discriminator.trainable = False

        g_loss = gan_model.train_on_batch(noise, y_gan)

        d_losses.append(d_loss)
        g_losses.append(g_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

    return d_losses, g_losses


# STEP 8: INITIALIZE MODELS

latent_dim = 100

generator = build_generator(latent_dim)
discriminator = build_discriminator()
gan_model = build_gan(generator, discriminator, latent_dim)


# STEP 9: TRAIN MODEL

print("Training GAN...")
d_losses, g_losses = train_gan(generator, discriminator, gan_model,
                              latent_dim, epochs=1000, batch_size=64)


# STEP 10: GENERATE IMAGES

noise = generate_latent_points(latent_dim, 25)
generated_images = generator.predict(noise, verbose=0)

# Convert back to [0,1]
generated_images = (generated_images + 1) / 2.0

plt.figure(figsize=(5, 5))

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    plt.axis('off')

plt.suptitle("Generated MNIST Digits")
plt.show()


# STEP 11: PLOT LOSS GRAPH

plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training Performance')
plt.legend()
plt.show()
