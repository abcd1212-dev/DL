import numpy as np

# Sigmoid activation function → converts values to 00range (0,1)
sig = lambda x: 1/(1+np.exp(-x))

# Derivative of sigmoid → used in backpropagation
dsig = lambda x: x*(1-x)

# XOR input dataset (4 combinations)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Expected output
y =  np.array([[0],
              [1],
              [1],
              [0]])

# Initialize weights randomly
np.random.seed(1)

w1 = np.random.rand(2,2)   # w1 → weights from input → hidden layer
w2 = np.random.rand(2,1)   # w2 → weights from hidden → output layer

# Training loop
for _ in range(10000):

    # -------- FORWARD PROPAGATION --------

    h = sig(X @ w1)        # h → hidden layer output
                           # Input X multiplied by weights w1

    o = sig(h @ w2)        # o → final output (prediction)
                           # Hidden output multiplied by w2

    # -------- BACKPROPAGATION --------

    d2 = (y - o) * dsig(o)   # d2 → output layer error
                             # difference between actual (y) and predicted (o)

    d1 = d2 @ w2.T * dsig(h) # d1 → hidden layer error
                             # backpropagated error from output layer,@ wordis matrix Multiplication

    # -------- WEIGHT UPDATE --------

    w2 += h.T @ d2 * 0.1     # update weights from hidden → output
    w1 += X.T @ d1 * 0.1     # update weights from input → hidden


# Final prediction
print("Final Output:")
print(o)
