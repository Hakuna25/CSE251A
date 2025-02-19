import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import log_loss

# Load the wine dataset
data = load_wine()
X = data.data  # Features (178, 13)
y = data.target  # Labels (178,)

# Filter the dataset to keep only the first two classes (binary classification)
X = X[:130]
y = y[:130]

# Add a bias term (intercept) to the feature matrix
X = np.hstack([X, np.ones((X.shape[0], 1))])

# Initialize weights (including bias)
d = X.shape[1]  # Number of features (including bias)
w = np.zeros(d)

# Define the stabilized sigmoid function
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Clip z to avoid overflow
    return 1 / (1 + np.exp(-z))

# Define the loss function (log loss)
def compute_loss(X, y, w):
    y_pred = sigmoid(X @ w)
    return log_loss(y, y_pred)

# Define the gradient of the loss with respect to w
def compute_gradient(X, y, w):
    y_pred = sigmoid(X @ w)
    return X.T @ (y_pred - y) / len(y)

# Coordinate descent parameters
max_iter = 1000  # Maximum number of iterations
eta = 0.01  # Learning rate
tol = 1e-5  # Convergence tolerance
loss_history = []

# Coordinate descent algorithm
for iteration in range(max_iter):
    # Compute the gradient
    gradient = compute_gradient(X, y, w)
    
    # Choose the coordinate with the largest absolute gradient
    i = np.argmax(np.abs(gradient))
    
    # Update the selected coordinate
    w[i] -= eta * gradient[i]
    
    # Compute the loss and store it
    loss = compute_loss(X, y, w)
    loss_history.append(loss)
    
    # Check for convergence
    if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
        break

# Print the final loss
print(f"Final loss: {loss_history[-1]:.4f}")
print(f"Final weights: {w}")