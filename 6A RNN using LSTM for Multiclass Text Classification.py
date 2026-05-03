# IMPORT LIBRARIES

# TensorFlow is the main deep learning framework used to build and train models
import tensorflow as tf

# Reuters dataset is a preprocessed dataset for news classification
# Each sample is a sequence of word indices and a label (0–45)
from tensorflow.keras.datasets import reuters

# pad_sequences is used to make all input sequences the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sequential model allows stacking layers one after another
from tensorflow.keras.models import Sequential

# Layers used:
# Embedding → converts words to vectors
# LSTM → processes sequence data
# Dense → fully connected layers for classification
from tensorflow.keras.layers import Embedding, LSTM, Dense


# LOAD DATASET

# num_words=10000 means:
# Only the top 10,000 most frequent words are kept
# Less frequent words are ignored to reduce complexity
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=10000)


# DATA UNDERSTANDING

# Total number of training samples
print("Training samples:", len(X_train))

# Total number of testing samples
print("Test samples:", len(X_test))

# Show first review (only first 10 words for readability)
# These numbers represent word indices, not actual words
print("Example sequence:", X_train[0][:10])

# Label of that review (class between 0 and 45)
print("Example label:", y_train[0])


# PAD SEQUENCES

# Neural networks require fixed-length input
# max_len defines the length of each sequence
max_len = 200

# pad_sequences ensures all sequences have same length:
# - If sequence is shorter → pad with zeros at the end
# - If sequence is longer → truncate to 200 words
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test  = pad_sequences(X_test, maxlen=max_len, padding='post')


# BUILD LSTM MODEL

# Sequential model processes data layer by layer
model = Sequential([

    # Embedding Layer
    # Converts integer word indices into dense vectors
    # input_dim=10000 → vocabulary size
    # output_dim=64 → each word becomes a 64-dimensional vector
    # input_length=max_len → length of each input sequence
    Embedding(input_dim=10000, output_dim=64, input_length=max_len),

    # LSTM Layer
    # Processes sequence step-by-step and keeps memory of previous words
    # Useful for understanding context in text data
    # 64 units → size of hidden memory state
    LSTM(64),

    # Dense Layer
    # Fully connected layer that learns complex patterns from LSTM output
    # ReLU activation helps in learning non-linear relationships
    Dense(32, activation='relu'),

    # Output Layer
    # 46 neurons because there are 46 classes in Reuters dataset
    # Softmax converts outputs into probability distribution
    Dense(46, activation='softmax')
])


# COMPILE MODEL

# optimizer='adam':
# Efficient algorithm to update weights during training

# loss='sparse_categorical_crossentropy':
# Used because labels are integers (not one-hot encoded)

# metrics=['accuracy']:
# Measures how many predictions are correct
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# TRAIN MODEL

# epochs=15:
# The model will go through the entire dataset 15 times

# validation_split=0.2:
# 20% of training data is used for validation
# Helps monitor performance on unseen data during training
history = model.fit(
    X_train, y_train,
    epochs=15,
    validation_split=0.2
)


# EVALUATE MODEL

# Evaluate model on test data (completely unseen during training)
# Returns loss and accuracy
loss, acc = model.evaluate(X_test, y_test)

# Print final accuracy
print("LSTM Test Accuracy:", acc)
