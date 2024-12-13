# Empirical Analysis of neural network training using data parallelism

# Introduction 

The growing volume of accumulated data in recent times presents both opportunities and challenges for data scientists. On the positive side, larger datasets allow for training Machine Learning (ML) algorithms more effectively, leading to improved predictive accuracy. However, the downside is that increased data volume significantly extends training times, limiting the number of hypotheses that can be tested due to the prolonged training process. For deep learning models in particular, there are two approaches to attain parallelism: Data Parallelism and Model Parallelism. 

## Data Parallelism 
In data parallelism, the dataset is split into smaller batches, each assigned to a separate computing unit. Each computing unit processes its batch using an identical copy of the model, calculating the loss (measuring prediction accuracy) and gradients (indicating how to adjust model weights). Once all the processors complete their tasks, the gradients are aggregated to update the model’s weights, preparing it for the next learning iteration. This approach enables efficient training of large datasets by distributing computation across devices without altering the model itself. 

<img width="676" alt="Screenshot 2024-12-05 at 6 07 47 PM" src="https://github.com/user-attachments/assets/af990799-2064-4729-a0e7-0a1cf126ad96">

## Model parallelism

In model parallelism, the model is split across processing units, with different parts (e.g., layers or neuron groups) assigned to different devices. Computation proceeds sequentially, requiring intermediate outputs (activations) to be transferred between processing units as data flows through the model. For example, in a simple neural network with a hidden layer and an output layer, the hidden layer might be processed on one unit, while the output layer is handled by another. This approach is valuable for models too large to fit into a single processor's memory.

<img width="702" alt="Screenshot 2024-12-05 at 6 08 20 PM" src="https://github.com/user-attachments/assets/3bbfad0b-a9f7-4de8-935c-7e4a6c1777bf">


## Scope of the project

This project explores the empirical training time performance of neural networks using traditional sequential training and data parallelism with MPI. By leveraging data parallelism, the study assesses the efficiency of distributing data across multiple CPU cores. Given the complexity of implementing model parallelism, the scope is focused on data parallelism, where identical models process separate data batches. This approach provides a detailed evaluation of training times under both configurations and offers valuable insights into their practical applications in machine learning.

## Experimentation

To investigate the impact of data parallelism on training time, I designed an experiment using the MNIST dataset, a popular benchmark for machine learning models. I implemented a custom data generator that allows selecting subsets of the dataset, ranging from 10,000 samples to the entire dataset of 70,000 samples. The MNIST data, originally consisting of 784 features (28x28 images), was reduced to 400 dimensions using Principal Component Analysis (PCA) to optimize computational performance. Additionally, the data was normalized for consistency. The dataset was then saved in a .mat file for efficient loading and usage during training. This flexible setup enabled a detailed comparison of training times across varying dataset sizes, providing insight into the scalability of the implemented approach. The data preparation code is detailed above for reproducibility.

For the experiment, I utilized data parallelism with four processes, a limitation imposed by my personal laptop's memory capacity. The comparison was performed between training the model with and without parallelism across all dataset sizes, aiming to highlight the potential benefits of leveraging parallel processing for large-scale data.

## Findings 

The graph illustrates the training time comparison between executing a task with and without data parallelism across varying dataset sizes. As observed, the training time increases consistently with dataset size for both methods. However, the use of data parallelism significantly reduces the training time across all dataset sizes. Without parallelism, the training time exhibits a steeper growth rate, demonstrating higher computational cost as dataset size increases. In contrast, the curve representing parallel execution grows at a slower rate, indicating improved scalability and efficiency. The gap between the two lines widens with larger datasets, emphasizing the advantage of parallelism in handling computationally intensive tasks for large-scale datasets. This comparison clearly highlights the performance benefits of data parallelism in reducing training time and improving processing efficiency, especially for extensive datasets.

<img width="988" alt="Screenshot 2024-12-11 at 11 01 37 PM" src="https://github.com/user-attachments/assets/811007d2-7e24-414c-9360-ace3e1f1de9d" />



The following images illustrate the time taken per epoch for varying dataset sizes across both neural networks, comparing training with and without data parallelism.

<img width="990" alt="Screenshot 2024-12-12 at 11 13 46 PM" src="https://github.com/user-attachments/assets/f58c6412-1684-40ee-8dbc-50d86bf5d085" />




<img width="994" alt="Screenshot 2024-12-12 at 11 14 06 PM" src="https://github.com/user-attachments/assets/80829815-3aaf-4303-8183-fc8b2086a0bc" />




<img width="835" alt="Screenshot 2024-12-12 at 11 14 18 PM" src="https://github.com/user-attachments/assets/5cb93433-7103-41e6-a011-4512b1860f31" />




## Try it yourself!

1. Create a virtual environment and install the necessary packages by using the commmand "pip install -r requirements.txt". I used Python 3.11 for development, but you can use other versions, provided there are no compatibility issues between the Python version and the package versions.
2. To generate the dataset run the data_generation.py code using the command "python data_generation.py" . Note that you can change the size of the dataset here by changing the values on line 15 and 16.
3. To initialize the weights of the neural network, run the nn_weight_intializer.py file using the command "python nn_weight_intializer.py"
   4.a. Now to train the model in distributed mode, use the command "MNISTNN_PARALLEL=yes mpiexec -n 4 python model_training.py"
   4.b. To train the model without data parallelism, use the command "MNISTNN_PARALLEL=no python model_training.py"
   (Note : There are multiple alternatives to these two commands, I have just mentioned one for documentation purposes)


## Future Work

An interesting direction for future work would be exploring meta-learners, where each processing unit is assigned a different model to train. This approach allows the simultaneous training of diverse models, avoiding the need for sequential training and enabling the exploitation of the strengths of multiple models at once. The results from these models could then be aggregated to improve overall performance. This would not only enhance efficiency but also provide valuable insights into how combining different model architectures could benefit training processes in large-scale machine learning tasks.

