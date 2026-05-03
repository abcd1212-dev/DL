# IMPORT LIBRARIES
import tensorflow as tf

# Keras layers and models → high-level API to define neural network architecture
from tensorflow.keras import layers, models

# pad_sequences → ensures all input sequences have the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences

# IMDB dataset → preprocessed dataset of movie reviews for sentiment analysis
from tensorflow.keras.datasets import imdb

# NumPy → used for numerical operations and array manipulation
import numpy as np

# string → used to remove punctuation during text preprocessing
import string


# LOAD DATASET

# num_words=10000 means:
# Only the top 10,000 most frequent words are kept
# Less frequent words are ignored → reduces complexity and noise
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Each review is already converted into a sequence of integers
# Each integer represents a word index in the vocabulary


# DATASET UNDERSTANDING

# Print basic information about dataset
print("Training samples:", len(X_train))   # number of training reviews
print("Test samples:", len(X_test))        # number of test reviews

# Display first review (only first 10 words for readability)
print("Example review (encoded):", X_train[0][:10])

# Display its label (0 = negative, 1 = positive)
print("Example label:", y_train[0])


# PAD SEQUENCES

# Reviews have variable lengths, but neural networks require fixed-size input
# max_len defines the length of each input sequence
max_len = 200

# pad_sequences does two things:
# - Short reviews → padded with zeros at the beginning
# - Long reviews → truncated to the last 200 words
X_train = pad_sequences(X_train, maxlen=max_len)
X_test  = pad_sequences(X_test, maxlen=max_len)


# BUILD RNN MODEL

# Sequential model → layers are arranged in order (input → output)
model = models.Sequential([

    # Embedding Layer
    # Converts word indices into dense vectors (word embeddings)
    # input_dim=10000 → vocabulary size
    # output_dim=32 → each word is represented as a 32-dimensional vector
    # input_length=max_len → length of each review
    layers.Embedding(input_dim=10000, output_dim=32, input_length=max_len),

    # SimpleRNN Layer
    # Processes the sequence step-by-step
    # Maintains a hidden state that captures information from previous words
    # 32 units → size of this hidden state
    layers.SimpleRNN(32),

    # Dense Layer
    # Fully connected layer that learns patterns from RNN output
    # ReLU activation helps in learning complex relationships
    layers.Dense(16, activation='relu'),

    # Output Layer
    # Single neuron because this is binary classification
    # Sigmoid activation outputs probability between 0 and 1
    layers.Dense(1, activation='sigmoid')
])


# MODEL SUMMARY

# Displays:
# - Layer types
# - Output shapes
# - Number of trainable parameters
# Helps understand model complexity
model.summary()


# COMPILE MODEL

# optimizer='adam':
# Adjusts learning rate automatically → faster convergence

# loss='binary_crossentropy':
# Measures error for binary classification problems

# metrics=['accuracy']:
# Tracks how many predictions are correct
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# TRAIN MODEL

# epochs=15:
# Model sees the entire dataset 15 times → improves learning

# validation_split=0.2:
# 20% of training data is used for validation
# Helps monitor performance on unseen data during training
history = model.fit(
    X_train, y_train,
    epochs=15,
    validation_split=0.2
)


# EVALUATE MODEL

# Evaluates model performance on completely unseen test data
# Returns loss and accuracy
loss, acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", acc)


# CUSTOM PREDICTION

# Get mapping of words to indices
word_index = imdb.get_word_index()

# Function to preprocess a new review
def preprocess_review(text):

    # Convert text to lowercase and remove punctuation
    # This ensures consistency with training data
    text = text.lower().translate(str.maketrans('', '', string.punctuation))

    # Split sentence into words
    words = text.split()

    # Convert each word into its corresponding index
    # If word is not found → assign index 2 (unknown token)
    encoded = [word_index.get(word, 2) + 3 for word in words]

    # Pad sequence to match model input size
    padded = pad_sequences([encoded], maxlen=max_len)

    return padded


# Example review for testing
review = "This movie was really good and enjoyable"

# Preprocess the review
processed = preprocess_review(review)

# Predict sentiment probability
prediction = model.predict(processed)[0][0]

# Convert probability into label
sentiment = "Positive" if prediction >= 0.5 else "Negative"

print("\nReview:", review)
print("Prediction:", prediction)
print("Sentiment:", sentiment)
