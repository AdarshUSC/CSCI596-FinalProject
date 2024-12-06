
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import CategoricalAccuracy

import tensorflow as tf
import os
import time
import sys


# # Configure GPUs for memory growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# tf.config.list_physical_devices('GPU')

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshape and normalize the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Split the test data into validation and test sets
val_size = int(0.2 * len(x_test))
x_val, y_val = x_test[:val_size], y_test[:val_size]
x_test, y_test = x_test[val_size:], y_test[val_size:]


log_file = "training_sequential_log.txt"
# sys.stdout = open(log_file, "w")

logdir = 'logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

start_time = time.time()

# Train the model
hist = model.fit(
    x_train, y_train,
    epochs=1,
    validation_data=(x_val, y_val),
    callbacks=[tensorboard_callback]
)

end_time = time.time()
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

# sys.stdout.close()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy}")


model.save(os.path.join('models', 'fashion_mnist_classifier.h5'))
