import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Load IMDB dataset (binary sentiment classification)
# - Reviews labeled as 0 (negative) or 1 (positive)
# - num_words=10000 → keep top 10,000 most frequent words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Pad sequences
# - Reviews have different lengths → RNN needs fixed length
# - pad/truncate all sequences to length 200
X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)

# Build RNN model
model = models.Sequential([

    # Embedding Layer
    # - Converts word indices → dense vectors
    # - Each word represented as 32-dimensional vector
    layers.Embedding(input_dim=10000, output_dim=32, input_length=200),

    # Simple RNN Layer
    # - Processes sequence step-by-step (captures order of words)
    # - 32 units → size of hidden state
    layers.SimpleRNN(32),

    # Fully connected layer
    # - Learns final decision from RNN output
    layers.Dense(16, activation='relu'),

    # Output layer
    # - 1 neuron → binary classification (0 or 1)
    # - Sigmoid outputs probability between 0 and 1
    layers.Dense(1, activation='sigmoid')
])

# Compile model
# - binary_crossentropy → used for binary classification
# - adam → efficient optimizer
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
# - epochs=5 → passes dataset 5 times
# - validation_data → checks performance on unseen data
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate model
print("Test Accuracy:", model.evaluate(X_test, y_test)[1])
