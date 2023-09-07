# Image Data Classification using Neural Networks

We implemented a multilayer perceptron (MLP) neural network from scratch in Python for image data classification. Here's what we did:

## Task 1: Acquire the Data
- Obtained the Fashion-MNIST dataset.
- Utilized machine learning libraries to load and **preprocess** the data, including **vectorization** and **normalization**.

## Task 2: Implement a Multilayer Perceptron (MLP) Neural Network
- Implemented the MLP from scratch.
- Created a Python class for the MLP with customizable activation functions, hidden layers, and units.
- Implemented **backpropagation** and mini-batch **gradient descent** algorithms.
- Defined functions for training (fit) and prediction (predict).
- Developed an accuracy evaluation function (evaluateacc).

## Task 3: Run the Experiments and Report
- Conducted various experiments to explore the impact of different factors on model performance.
- Split the dataset into training and test sets.
- Completed a minimum set of experiments:
    1. Evaluated MLP models with different numbers of hidden layers and activations (ReLU, tanh, Leaky-ReLU).
    2. Compared the performance of these models.
    3. Added dropout regularization to an MLP and observed its effect on accuracy.
    4. Trained an MLP with unnormalized images and assessed its accuracy.
    5. Created a **convolutional neural network (CNN)** and compared its performance to our MLP model.
    6. Tuned an MLP architecture for optimal performance.

## Deliverables
- Submitted two files as specified:
    1. **Code**: Containing the implemented MLP & CNN code, training, and experiment code.
    2. **Report**: A scientific report discussing the contents of this project.
