import os
import time
import numpy as np
import scipy.io as sio
from mpi4py import MPI

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Configuration Flags
IS_DISTRIBUTED = os.getenv('MNISTNN_PARALLEL') == 'yes'

# Neural Network Parameters
INPUT_SIZE = 400
HIDDEN_SIZE = 50
OUTPUT_SIZE = 10

# Data Loading Function
def load_data(file='mnistdata.mat'):
    data = sio.loadmat(file)
    return data['X'].astype('f8'), data['y'].reshape(-1)

# Initialize Neural Network Weights
def init_weights(input_size, output_size):
    epsilon = 0.12
    return np.random.uniform(-epsilon, epsilon, (output_size, input_size + 1))

# Activation Functions
def activation_sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_sigmoid(z):
    sig = activation_sigmoid(z)
    return sig * (1 - sig)

# Compute Cost and Gradients
def calculate_cost_and_gradients(theta1, theta2, features, targets):
    m = features.shape[0]

    # Forward Pass
    bias_features = np.insert(features, 0, 1, axis=1)
    hidden_input = np.dot(bias_features, theta1.T)
    hidden_output = np.insert(activation_sigmoid(hidden_input), 0, 1, axis=1)
    final_input = np.dot(hidden_output, theta2.T)
    final_output = activation_sigmoid(final_input)

    # Encode Labels
    encoded_labels = np.zeros((m, OUTPUT_SIZE))
    encoded_labels[np.arange(m), targets - 1] = 1

    # Compute Cost
    cost = (-1 / m) * np.sum(encoded_labels * np.log(final_output) + (1 - encoded_labels) * np.log(1 - final_output))

    # Backpropagation
    delta3 = final_output - encoded_labels
    delta2 = (np.dot(delta3, theta2) * gradient_sigmoid(np.insert(hidden_input, 0, 1, axis=1)))[:, 1:]

    theta1_grad = np.dot(delta2.T, bias_features) / m
    theta2_grad = np.dot(delta3.T, hidden_output) / m

    return cost, (theta1_grad, theta2_grad)

# Training Routine
def train_nn(features, targets, learning_rate=0.1, max_iterations=50, epochs=5):
    theta1 = init_weights(INPUT_SIZE, HIDDEN_SIZE)
    theta2 = init_weights(HIDDEN_SIZE, OUTPUT_SIZE)

    if IS_DISTRIBUTED:
        feature_splits = np.array_split(features, size)
        target_splits = np.array_split(targets, size)
        local_features = feature_splits[rank]
        local_targets = target_splits[rank]
    else:
        local_features = features
        local_targets = targets

    start_time = time.time()

    for epoch in range(epochs):
        if rank == 0:
            print(f"Starting epoch {epoch + 1} of {epochs}")
        epoch_start_time = time.time()

        for _ in range(max_iterations):
            cost, (local_grad1, local_grad2) = calculate_cost_and_gradients(theta1, theta2, local_features, local_targets)

            if IS_DISTRIBUTED:
                gathered_grad1 = comm.gather(local_grad1, root=0)
                gathered_grad2 = comm.gather(local_grad2, root=0)

                if rank == 0:
                    grad1 = np.mean(gathered_grad1, axis=0)
                    grad2 = np.mean(gathered_grad2, axis=0)

                    theta1 -= learning_rate * grad1
                    theta2 -= learning_rate * grad2

                theta1 = comm.bcast(theta1, root=0)
                theta2 = comm.bcast(theta2, root=0)
            else:
                theta1 -= learning_rate * local_grad1
                theta2 -= learning_rate * local_grad2

        if rank == 0:
            epoch_end_time = time.time()
            print(f"Epoch {epoch + 1} completed in {epoch_end_time - epoch_start_time:.2f} seconds.")

    if rank == 0:
        print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    return theta1, theta2

# Prediction Function
def make_predictions(model, features):
    theta1, theta2 = model
    features_with_bias = np.insert(features, 0, 1, axis=1)
    hidden_output = np.insert(activation_sigmoid(np.dot(features_with_bias, theta1.T)), 0, 1, axis=1)
    final_output = activation_sigmoid(np.dot(hidden_output, theta2.T))
    return np.argmax(final_output, axis=1) + 1

# Accuracy Calculation
def compute_accuracy(actual, predicted):
    return np.mean(actual == predicted) * 100

# Main Execution
def main():
    features, targets = load_data()

    if rank == 0:
        print(f"Mode: {'Distributed' if IS_DISTRIBUTED else 'Standalone'}")
        print(f"Dataset size: {len(features)}, Processes: {size}")

    model = train_nn(features, targets, learning_rate=0.1, max_iterations=100, epochs=10)

    if rank == 0:
        predictions = make_predictions(model, features)
        accuracy = compute_accuracy(targets, predictions)
        print(f"Training Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
