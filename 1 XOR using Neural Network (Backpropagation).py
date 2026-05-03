#XOR 
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




# AND Gate (Perceptron)
import numpy as np  # used for consistency (not really needed here)

# Step activation function
# Returns 1 if input >= 0, otherwise 0
step = lambda x: 1 if x >= 0 else 0

# Function to implement AND gate
def AND(x1, x2):
    w1, w2 = 1, 1        # weights for both inputs
    bias = -1.5          # bias shifts decision boundary

    # weighted sum (net input)
    net = x1*w1 + x2*w2 + bias

    # apply activation function
    return step(net)

# Testing all input combinations
for x1 in [0,1]:
    for x2 in [0,1]:
        print(x1, x2, "->", AND(x1,x2))  # print result


#OR Gate Perceptron

import numpy as np

# Step activation function
step = lambda x: 1 if x >= 0 else 0

# Function to implement OR gate
def OR(x1, x2):
    w1, w2 = 1, 1        # weights
    bias = -0.5          # lower threshold than AND

    # weighted sum
    net = x1*w1 + x2*w2 + bias

    # activation
    return step(net)

# Testing all combinations
for x1 in [0,1]:
    for x2 in [0,1]:
        print(x1, x2, "->", OR(x1,x2))
