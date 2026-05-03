# IMPORT LIBRARIES

# TensorFlow is used for building and training deep learning models
import tensorflow as tf

# Reuters dataset for multiclass text classification (46 categories)
from tensorflow.keras.datasets import reuters

# pad_sequences is used to make all sequences the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sequential model allows stacking layers in order
from tensorflow.keras.models import Sequential

# GRU is the recurrent layer used here (alternative to LSTM)
# Embedding converts words to vectors, Dense for classification
from tensorflow.keras.layers import Embedding, GRU, Dense


# LOAD DATASET

# num_words=10000 → keep top 10,000 most frequent words
# Each input is a sequence of word indices
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=10000)


# DATA UNDERSTANDING

# Number of training samples
print("Training samples:", len(X_train))

# Number of testing samples
print("Test samples:", len(X_test))

# Example input sequence (first 10 words)
print("Example sequence:", X_train[0][:10])

# Corresponding label (0–45)
print("Example label:", y_train[0])


# PAD SEQUENCES

# Define fixed length for all sequences
max_len = 200

# pad_sequences ensures uniform input size:
# - Short sequences → padded with zeros at end
# - Long sequences → truncated
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test  = pad_sequences(X_test, maxlen=max_len, padding='post')


# BUILD GRU MODEL

# Sequential model processes layers step-by-step
model = Sequential([

    # Embedding Layer
    # Converts integer word indices into dense vectors
    # input_dim=10000 → vocabulary size
    # output_dim=64 → each word is a 64-dimensional vector
    Embedding(input_dim=10000, output_dim=64, input_length=max_len),

    # GRU Layer
    # Processes sequence data and captures context
    # Similar to LSTM but simpler and faster (fewer gates)
    # 64 units → size of hidden state
    GRU(64),

    # Dense Layer
    # Learns complex relationships from GRU output
    Dense(32, activation='relu'),

    # Output Layer
    # 46 neurons → one for each category
    # Softmax gives probability distribution across classes
    Dense(46, activation='softmax')
])


# COMPILE MODEL

# Adam optimizer → efficient weight updates
# Sparse categorical crossentropy → for integer labels
# Accuracy → performance metric
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# TRAIN MODEL

# epochs=15 → dataset is passed 15 times
# validation_split=0.2 → 20% of training data used for validation
history = model.fit(
    X_train, y_train,
    epochs=15,
    validation_split=0.2
)


# EVALUATE MODEL

# Evaluate model on unseen test data
loss, acc = model.evaluate(X_test, y_test)

# Print final accuracy
print("GRU Test Accuracy:", acc)
