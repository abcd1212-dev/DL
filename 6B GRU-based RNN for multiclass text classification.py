import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

# Load dataset
# - 46 topic classes
# - num_words=10000 keeps most frequent words
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=10000)

# Pad sequences to fixed length
# - Ensures all inputs have same size for GRU
X_train = pad_sequences(X_train, maxlen=200, padding='post')
X_test = pad_sequences(X_test, maxlen=200, padding='post')

# Build GRU-based RNN model
model = Sequential([
    
    # Embedding layer
    # - Converts word indices into dense vector representations
    Embedding(input_dim=10000, output_dim=64, input_length=200),

    # GRU layer
    # - Processes sequence data while keeping memory of past words
    # - Simpler and faster than LSTM but still captures context
    GRU(64),

    # Fully connected layer (optional feature refinement)
    Dense(32, activation='relu'),

    # Output layer
    # - 46 neurons for 46 classes
    # - softmax gives probability distribution
    Dense(46, activation='softmax')
])

# Compile model
# - sparse_categorical_crossentropy used because labels are integers (0–45)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train,
          epochs=5,
          batch_size=64,
          validation_data=(X_test, y_test))

# Evaluate model
print("Test Accuracy:", model.evaluate(X_test, y_test)[1])
