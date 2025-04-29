import numpy as np
import math
import random

# Helper Functions
def softmax(x):
    exps = np.exp(x - np.max(x))  # Stability improvement
    return exps / np.sum(exps)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def one_hot(index, size):
    vec = np.zeros(size)
    vec[index] = 1
    return vec

def cross_entropy(predicted, target_index):
    return -np.log(predicted[target_index] + 1e-10)

def initialize_weights(rows, cols):
    return np.random.uniform(-0.1, 0.1, (rows, cols))

# Training Data
word_to_idx = {"the": 0, "cat": 1, "sat": 2, "mat": 3}
idx_to_word = {i: w for w, i in word_to_idx.items()}
sentence = ["the", "cat", "sat", "mat"]
inputs = sentence[:-1]
target = sentence[-1]

input_size = len(word_to_idx)
hidden_size = 5  # Increased for experimentation
output_size = len(word_to_idx)

# One-hot encoding
input_vectors = [one_hot(word_to_idx[word], input_size) for word in inputs]
target_vector = one_hot(word_to_idx[target], output_size)

# Initialize Weights and Biases
Wxh = initialize_weights(hidden_size, input_size)
Whh = initialize_weights(hidden_size, hidden_size)
Why = initialize_weights(output_size, hidden_size)
bh = np.zeros(hidden_size)
by = np.zeros(output_size)

# Training Parameters
epochs = 50
learning_rate = 0.1

# Training Loop
for epoch in range(epochs):
    # Forward Pass
    h_prev = np.zeros(hidden_size)
    for x in input_vectors:
        h_current = tanh(np.dot(Wxh, x) + np.dot(Whh, h_prev) + bh)
        h_prev = h_current

    logits = np.dot(Why, h_current) + by
    y_pred = softmax(logits)

    # Loss Calculation
    target_index = word_to_idx[target]
    loss = cross_entropy(y_pred, target_index)

    # Backward Pass
    dy = y_pred.copy()
    dy[target_index] -= 1

    dWhy = np.outer(dy, h_current)
    dby = dy

    dh = np.dot(Why.T, dy) * tanh_derivative(h_current)

    dWxh = np.outer(dh, input_vectors[-1])  # Only last input matters for prediction
    dWhh = np.outer(dh, h_prev)
    dbh = dh

    # Update Weights and Biases
    Wxh -= learning_rate * dWxh
    Whh -= learning_rate * dWhh
    Why -= learning_rate * dWhy
    bh -= learning_rate * dbh
    by -= learning_rate * dby

    # Log Progress
    if epoch % 10 == 0 or epoch == epochs - 1:
        predicted_index = np.argmax(y_pred)
        predicted_word = idx_to_word[predicted_index]
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.4f} | Prediction: {predicted_word}")

# Final Output
predicted_index = np.argmax(y_pred)
predicted_word = idx_to_word[predicted_index]
print("\nFinal Prediction:")
print(f"Predicted word: {predicted_word}")
print(f"Target word: {target}")
