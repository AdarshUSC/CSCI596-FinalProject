import tensorflow as tf
import os
from mpi4py import MPI
import shutil

# MPI Configuration
comm = MPI.COMM_WORLD
NODES = comm.Get_size()
RANK = comm.Get_rank()

# Directories
DIR = "cumulative_data"  # Source directory for raw data (will be created)
OUTPUT_DIR = "distributed_data"  # Output directory for separated data (per rank)

if RANK == 0:
    # Load Fashion MNIST dataset
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (_, _) = fashion_mnist.load_data()

    # Normalize images and reshape them to (28, 28, 1)
    x_train = x_train.astype('uint8')
    x_train = x_train[..., None]  # Add channel dimension

    # Ensure the augmented directory exists
    if not os.path.exists(DIR):
        os.mkdir(DIR)

    # Create subdirectories for each class (0–9)
    for class_idx in range(10):
        class_dir = os.path.join(DIR, str(class_idx))
        os.mkdir(class_dir)

    # Save each image in its corresponding class folder
    for idx, image in enumerate(x_train):
        class_dir = os.path.join(DIR, str(y_train[idx]))
        image_path = os.path.join(class_dir, f"{idx}.png")
        tf.keras.utils.save_img(image_path, image)

comm.Barrier()

# Separate data into NODES subfolders
if RANK == 0:
    os.mkdir(OUTPUT_DIR)

comm.Barrier()

for i in range(NODES):
    if i == RANK:
        print("Rank: ", RANK)
        dir1 = os.path.join(OUTPUT_DIR, f"data{RANK + 1}")
        os.mkdir(dir1)

        classes = os.listdir(DIR)
        for class_idx in classes:
            src_class_dir = os.path.join(DIR, class_idx)
            dest_class_dir = os.path.join(dir1, class_idx)
            os.mkdir(dest_class_dir)

            # List all images in the source class directory
            img_paths = sorted(os.listdir(src_class_dir))
            num_images = len(img_paths)

            # Compute image allocation per node
            qt = num_images // NODES
            rem = num_images % NODES

            start = i * qt
            end = start + qt

            # Copy assigned images to the destination directory
            for k in range(start, end):
                src_path = os.path.join(src_class_dir, img_paths[k])
                shutil.copy(src_path, dest_class_dir)

            # Node 0 handles any remainder images
            if i == 0:
                for k in range(-rem, 0):
                    src_path = os.path.join(src_class_dir, img_paths[k])
                    shutil.copy(src_path, dest_class_dir)
