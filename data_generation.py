import numpy as np
import scipy.io as sio
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def prepare_mnist_data():
    # Fetch MNIST data
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target
#     print(len(X))

    y = y.astype(int) + 1

    X = X[:70000]
    y = y[:70000]

    # Reduce feature dimensions to 400 (20x20 pixels)
    
    pca = PCA(n_components=400)
    X_reduced = pca.fit_transform(X)

    # Optional: Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    # Create .mat file
    sio.savemat('mnistdata.mat', {
        'X': X_scaled,
        'y': y.reshape(-1, 1)
    })

    print("MNIST data saved to mnistdata.mat")
    print(f"X shape: {X_scaled.shape}")
    print(f"y shape: {y.shape}")
    
    visualize_images(X, y)
    

def visualize_images(X, y, num_images=10):
    """
    Visualizes a few images from the MNIST dataset.
    """
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')  # Reshape each sample back to 28x28
        plt.title(f"Label: {y[i]-1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    prepare_mnist_data()
