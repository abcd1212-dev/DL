# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset (handwritten digits)
# We ignore labels because GAN is unsupervised
(X_train, _), _ = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from [0, 255] to [-1, 1]
# This helps the generator learn more stable outputs
X_train = (X_train - 127.5) / 127.5

# Reshape data to include channel dimension (needed for neural network)
X_train = X_train.reshape(-1, 28, 28, 1)

# Generator model
# Takes random noise (100-d vector) and generates a fake image
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(100,)),
        layers.Dense(784, activation='tanh'),  # output flattened image
        layers.Reshape((28, 28, 1))            # reshape into image format
    ])
    return model

# Discriminator model
# Takes image and decides if it is real (1) or fake (0)
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # probability output
    ])
    return model

# Create models
generator = build_generator()
discriminator = build_discriminator()

# Compile discriminator separately
# It learns to classify real vs fake images
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# Build GAN by stacking generator and discriminator
# During GAN training, discriminator weights are frozen
discriminator.trainable = False

gan_input = tf.keras.Input(shape=(100,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)

gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training loop
epochs = 1000
batch_size = 32

for epoch in range(epochs):

    # Step 1: Train discriminator

    # Select random real images from dataset
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

    # Step 2: Train generator through GAN

    discriminator.trainable = False

    noise = np.random.normal(0, 1, (batch_size, 100))

    # Generator tries to fool discriminator (wants output = 1)
    g_loss = gan.train_on_batch(
        noise,
        np.ones((batch_size, 1))
    )

    # Print progress occasionally
    if epoch % 200 == 0:
        print("Epoch:", epoch, "D Loss:", d_loss_real, "G Loss:", g_loss)

# Generate a sample image after training
noise = np.random.normal(0, 1, (1, 100))
gen_img = generator.predict(noise, verbose=0)

plt.imshow(gen_img[0, :, :, 0], cmap='gray')
plt.title("Generated Image")
plt.show()
