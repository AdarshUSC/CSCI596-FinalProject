from mpi4py import MPI
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

# Initialize MPI communication
comm = MPI.COMM_WORLD
world_size = comm.Get_size()
node_rank = comm.Get_rank()

comm.Barrier()

if node_rank == 0:
    start_time = time.time()

# Function to synchronize and update model weights across nodes
def synchronize_model_weights(model):
    global_weights = []
    averaged_weights = []

    for layer in model.layers:
        # Skip non-trainable layers
        if 'flatten' not in layer.name and 'max_pooling' not in layer.name:
            if node_rank == 0:
                print("Processing on Master Node: ", layer.name)

                # Collect weights from worker nodes
                layer_weights = [
                    comm.recv(source=i, tag=10) for i in range(1, world_size)
                ]

                # Compute average weights
                averaged_weights = sum(layer_weights) / world_size
                averaged_weights = comm.bcast(averaged_weights, root=0)

                global_weights.append(averaged_weights)
            else:
                # Send local weights to the master node
                local_weights = layer.get_weights()[0]
                comm.send(local_weights, dest=0, tag=10)

                # Receive the updated weights
                averaged_weights = comm.bcast(averaged_weights, root=0)
                layer.set_weights([np.array(averaged_weights), layer.get_weights()[1]])

            comm.Barrier()

    return global_weights if node_rank == 0 else None

# Function to prepare data, build, and train the model
def train_model(dataset):
    # Split dataset into training, validation, and testing sets
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - val_size

    train_data = dataset.take(train_size)
    val_data = dataset.skip(train_size).take(val_size)
    test_data = dataset.skip(train_size + val_size)

    # Build the model
    cnn_model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')  # Softmax for multi-class classification
    ])

    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.summary()

    # Train the model for multiple iterations
    for iteration in range(5):
        print(f"Starting iteration {iteration + 1}")
        cnn_model.fit(train_data, epochs=1, validation_data=val_data)
        synchronize_model_weights(cnn_model)

# Execute training on assigned data partitions
if node_rank == 0:
    print("Master node is processing...")
    data_part = tf.keras.utils.image_dataset_from_directory("data/data1")
    train_model(data_part)

elif node_rank == 1:
    print("Worker Node 1 is processing...")
    data_part = tf.keras.utils.image_dataset_from_directory("data/data2")
    train_model(data_part)

elif node_rank == 2:
    print("Worker Node 2 is processing...")
    data_part = tf.keras.utils.image_dataset_from_directory("data/data3")
    train_model(data_part)

elif node_rank == 3:
    print("Worker Node 3 is processing...")
    data_part = tf.keras.utils.image_dataset_from_directory("data/data4")
    train_model(data_part)

comm.Barrier()

if node_rank == 0:
    end_time = time.time()
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")
