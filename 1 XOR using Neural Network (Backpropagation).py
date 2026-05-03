# Import NumPy library
# Used for arrays, matrix multiplication, and mathematical functions
import numpy as np


# Define Sigmoid Activation Function
# Formula = 1 / (1 + e^-x)
# Converts any input value into range 0 to 1
# Used to introduce non-linearity in neural network
def sig(x):
    return 1 / (1 + np.exp(-x))


# Derivative of Sigmoid Function
# Formula = x * (1 - x)
# Required during backpropagation to update weights
# Here x is already sigmoid output
def dsig(x):
    return x * (1 - x)


# XOR Input Dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])


# Expected XOR Output
# XOR gives 1 when inputs are different
# XOR gives 0 when inputs are same
y = np.array([[0],
              [1],
              [1],
              [0]])


# Set random seed
# Gives same random weights every time program runs
np.random.seed(1)


# Initialize first weight matrix
# Connects Input Layer to Hidden Layer
# 2 input neurons and 2 hidden neurons
# Shape = (2 x 2)
w1 = np.random.rand(2,2)


# Initialize second weight matrix
# Connects Hidden Layer to Output Layer
# 2 hidden neurons and 1 output neuron
# Shape = (2 x 1)
w2 = np.random.rand(2,1)


# Learning rate  Controls speed of weight updates
lr = 0.1


# Training loop
# Run network 10000 times to learn XOR pattern
for epoch in range(10000):

    # Hidden Layer Calculation
    # Multiply input matrix X with weights w1
    # Then apply sigmoid activation
    h = sig(X @ w1)

    # Output Layer Calculation
    # Multiply hidden output h with weights w2
    # Then apply sigmoid activation
    o = sig(h @ w2)

    # Output Layer Error
    # (Actual output - Predicted output)
    d2 = (y - o) * dsig(o)

    # Hidden Layer Error
    # Backpropagate error to hidden layer
    d1 = d2 @ w2.T * dsig(h)

    # Update weights (Hidden → Output)
    w2 += h.T @ d2 * lr

    # Update weights (Input → Hidden)
    w1 += X.T @ d1 * lr


# Print final predicted decimal outputs
print("\n===== XOR OUTPUT =====")
print(o)


# Convert decimal outputs into binary values
# If output >= 0.5 then 1 else 0
binary_output = (o >= 0.5).astype(int)

print("\nBinary Output:")
for i, val in enumerate(binary_output):
    print(X[i], "->", val[0])


# ================= McCulloch–Pitts Model =================

# Step activation function
# Returns 1 if input >= 0 else 0
step = lambda x: 1 if x >= 0 else 0


# McCulloch–Pitts AND Gate
# Uses fixed weights (no learning)
def MP_AND(x1, x2):
    w1, w2 = 1, 1        # weights
    bias = -1.5          # threshold

    # weighted sum
    net = x1*w1 + x2*w2 + bias

    # activation
    return step(net)


print("\n===== McCulloch–Pitts AND Gate =====")

for x1 in [0,1]:
    for x2 in [0,1]:
        print(x1, x2, "->", MP_AND(x1,x2))


# ================= Perceptron Model =================

# Expected AND Output
y_and = np.array([0, 0, 0, 1])


# Initialize weights and bias
# Random values (will be learned)
w = np.random.rand(2)
b = np.random.rand()


# Learning rate
lr = 0.1


# Training perceptron
# Updates weights based on error
for epoch in range(10):

    for i in range(len(X)):

        # weighted sum
        net = np.dot(X[i], w) + b

        # prediction using step function
        pred = step(net)

        # error calculation
        error = y_and[i] - pred

        # update weights
        w += lr * error * X[i]

        # update bias
        b += lr * error


print("\n===== Perceptron AND Gate =====")

for i in range(len(X)):
    net = np.dot(X[i], w) + b
    print(X[i], "->", step(net))


# ================= OR Gate =================

# OR Gate using step function
def OR(x1, x2):
    w1, w2 = 1, 1        # weights
    bias = -0.5          # threshold

    # weighted sum
    net = x1*w1 + x2*w2 + bias

    # activation
    return step(net)


print("\n===== OR GATE =====")

for x1 in [0,1]:
    for x2 in [0,1]:
        print(x1, x2, "->", OR(x1,x2))
