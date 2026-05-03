import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load Reuters dataset
# - News articles represented as sequences of word indices
# - 46 different topic categories (multiclass classification)
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=10000)

# Pad sequences to fixed length
# - Neural networks require uniform input size
# - maxlen=200 ensures all sequences are same length
X_train = pad_sequences(X_train, maxlen=200, padding='post')
X_test = pad_sequences(X_test, maxlen=200, padding='post')

# Build LSTM model
model = Sequential([
    
    # Embedding layer
    # - Converts word indices into dense 64-dimensional vectors
    # - Learns semantic relationships between words
    Embedding(input_dim=10000, output_dim=64, input_length=200),

    # LSTM layer
    # - Processes words in sequence order
    # - Captures context and long-term dependencies in text
    LSTM(64),

    # Output layer
    # - 46 neurons = 46 news categories
    # - Softmax gives probability distribution over classes
    Dense(46, activation='softmax')
])

# Compile model
# - adam: adaptive optimizer for efficient training
# - sparse_categorical_crossentropy: used for integer labels (0–45)
# - accuracy: performance metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
# - epochs=5: full dataset passed 5 times
# - batch_size=64: updates weights after every 64 samples
# - validation_data: evaluates model on unseen test data
model.fit(X_train, y_train,epochs=5,validation_data=(X_test, y_test))

# Evaluate model on test data
# - returns loss and accuracy
print("Test Accuracy:", model.evaluate(X_test, y_test)[1])
