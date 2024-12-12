import numpy as np
import scipy.io as sio

def generate_initial_weights():
    # Match the network structure in the original script
    Input_layer_size = 400
    Hidden_layer_size = 50
    Output_layer_size = 10

    # Random initialization with epsilon
    epsilon_init = 0.12

    # Generate Theta1 and Theta2 matrices
    Theta1 = np.random.rand(Hidden_layer_size, 1 + Input_layer_size) * 2 * epsilon_init - epsilon_init
    Theta2 = np.random.rand(Output_layer_size, 1 + Hidden_layer_size) * 2 * epsilon_init - epsilon_init

    # Save to .mat file
    sio.savemat('mnistweights.mat', {
        'Theta1': Theta1,
        'Theta2': Theta2
    })

    print("Initial weights saved to mnistweights.mat")
    print(f"Theta1 shape: {Theta1.shape}")
    print(f"Theta2 shape: {Theta2.shape}")

if __name__ == '__main__':
    generate_initial_weights()
