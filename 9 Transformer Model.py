# STEP 1: Import required libraries
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds


# STEP 2: Load dataset (AG News)

# split=['train','test'] → gives separate training & testing data
# as_supervised=True → returns (text, label) pairs
train_data, test_data = tfds.load(
    "ag_news_subset",
    split=['train', 'test'],
    as_supervised=True
)

# Batch data → groups data into batches of 32 samples
train_data = train_data.batch(32)
test_data = test_data.batch(32)


# STEP 3: Text Vectorization (Text → Numbers)

# max_tokens=10000 → keep top 10,000 words
# output_sequence_length=100 → each sentence becomes length 100
vectorize = layers.TextVectorization(
    max_tokens=10000,
    output_sequence_length=100
)

# Learn vocabulary from training data
vectorize.adapt(train_data.map(lambda x, y: x))

# STEP 4: Build Transformer Model

# Input layer → takes raw text (string)
inputs = tf.keras.Input(shape=(1,), dtype=tf.string)

# Convert text → integer tokens
x = vectorize(inputs)

# Embedding layer → converts tokens into dense vectors
# (word representation in vector form)
x = layers.Embedding(10000, 64)(x)

# Multi-Head Attention (core Transformer component)
# num_heads=2 → 2 attention heads
# key_dim=64 → size of attention vectors
x = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)

# Global pooling → reduces sequence into single vector
x = layers.GlobalAveragePooling1D()(x)

# Fully connected layer
x = layers.Dense(64, activation='relu')(x)

# Output layer → 4 classes (softmax for multiclass classification)
outputs = layers.Dense(4, activation='softmax')(x)

# Create model
model = tf.keras.Model(inputs, outputs)

# STEP 5: Compile mode

# optimizer='adam' → updates weights efficiently
# loss='sparse_categorical_crossentropy' → for multiclass labels
# metrics=['accuracy'] → evaluates performance
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# STEP 6: Train model

# epochs=3 → number of times model sees full dataset
model.fit(train_data, epochs=3, validation_data=test_data)

# STEP 7: Evaluate model

# Evaluate accuracy on test data
loss, acc = model.evaluate(test_data)

print("Test Accuracy:", acc)
