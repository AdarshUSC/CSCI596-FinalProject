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

This project investigates the empirical training time performance of neural networks using the typical sequential data training, data parallelism using MPI as well as training on a single GPU . By leveraging data parallelism, the study evaluates the efficiency of splitting data across multiple CPU cores while comparing it to the GPU's optimized training capabilities. Due to the complexity of implementing model parallelism, the focus is limited to data parallelism, where identical models process different data batches, providing a comprehensive evaluation of training time under both configurations and insights into their practical application in machine learning.

## Future Work

An interesting direction for future work would be exploring meta-learners, where each processing unit is assigned a different model to train. This approach allows the simultaneous training of diverse models, avoiding the need for sequential training and enabling the exploitation of the strengths of multiple models at once. The results from these models could then be aggregated to improve overall performance. This would not only enhance efficiency but also provide valuable insights into how combining different model architectures could benefit training processes in large-scale machine learning tasks.

