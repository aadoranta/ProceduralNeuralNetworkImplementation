import pandas as pd
import numpy as np

"""
SET UP AND PREPROCESSING
"""

# Hyper-parameters
learning_rate = 0.1
num_epochs = 50  # Number of times the network sees ALL the data
num_nodes = 25
split_ratio = 0.8


def activation(x):
    return 1 / (1 + np.exp(-x))


def d_activation(x):
    return activation(x) * (1 - activation(x))


# Data Processing
data = pd.read_csv("ANN - Iris data.txt", sep=',', header=None)
data.columns = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'label']

# Shuffle Data
data = data.sample(frac=1)
data_length = len(data)

# Get X and Y from data
Y = pd.get_dummies(data['label']).astype(int).to_numpy()
y_train = Y[0:int(data_length*split_ratio)]
y_test = Y[int(data_length*split_ratio):]

X = data[['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']].to_numpy()
x_train = X[0:int(data_length*split_ratio)]
x_test = X[int(data_length*split_ratio):]

# Instantiate initial weight matrices
first_matrix = np.random.rand(5, num_nodes) - 0.5
second_matrix = np.random.rand(num_nodes + 1, 3) - 0.5

# You can hardcode weight values if you want to see how everything works
# first_matrix = np.array([[0.5, 0.4],
#                          [0.4, 0.6],
#                          [-0.3, 0.2],
#                          [-0.9, -0.1],
#                          [0.5, 0.2]])
#
# second_matrix = np.array([[0.1, -0.20, 0.3],
#                           [-0.3, 0.1, 0.2],
#                           [-0.2, 0.4, -0.1]])

# Initialize metrics
correct = 0
incorrect = 0

"""
TRAINING LOOP
"""

for epoch in range(num_epochs):
    for x, y in zip(x_train, y_train):

        # You can hardcode an input to see all operations on a simple input
        # x = np.array([0.03, 0.04, 0.01, 0.05])

        """
        Forward Pass Section
        """

        # Append bias term
        x = np.append(x, 1)

        # Instantiate hidden output
        hidden_nodes = np.copy(first_matrix)

        # Perform matrix multiplication
        for i1, value in enumerate(x):
            hidden_nodes[i1] = hidden_nodes[i1] * value

        # Get the sum for each output node
        hidden_nodes = np.sum(hidden_nodes, axis=0)

        # Activation Function
        hidden_nodes = activation(hidden_nodes)

        # Add bias term
        hidden_nodes = np.append(hidden_nodes, 1)

        # Instantiate final output
        output = np.zeros_like(second_matrix)
        for i2, value in enumerate(hidden_nodes):
            output[i2] = np.multiply(second_matrix[i2], value)

        # Get the sum for each output node
        output = np.sum(output, axis=0)

        # Activation function
        output = activation(output)

        """
        Backward Pass Section
        """

        # Calculate the delta values for each output node
        output_delta = np.multiply(d_activation(output), (y - output))

        # Instantiate hidden delta vector
        hidden_delta = np.zeros_like(hidden_nodes)

        # Calculate delta values for hidden layer
        for i3, array in enumerate(second_matrix):
            hidden_delta[i3] = np.multiply(d_activation(hidden_nodes[i3]), np.sum(np.multiply(array, output_delta)))

        # Instantiate a matrix of weight updates for first layer weights
        first_matrix_updates = np.zeros_like(first_matrix)

        # Calculate updates to the first weight matrix
        # (We don't consider the last element of the hidden delta since it corresponds to a bias term)
        for i4, input_node in enumerate(x):
            first_matrix_updates[i4] = learning_rate * input_node * hidden_delta[:-1]

        # Update first weight matrix
        first_matrix = first_matrix + first_matrix_updates

        # Instantiate a matrix of weight updates for second layer weights
        second_matrix_updates = np.zeros_like(second_matrix)

        # Calculate updates to the second weight matrix
        for i5, hidden_output in enumerate(hidden_nodes):
            second_matrix_updates[i5] = learning_rate * hidden_output * output_delta

        # Update second weight matrix
        second_matrix = second_matrix + second_matrix_updates

        # Get accuracy metrics
        if epoch == num_epochs - 1:
            if np.argmax(output) == np.argmax(y):
                correct += 1
            else:
                incorrect += 1

# Report metrics
print("Training Accuracy: ", correct / (correct + incorrect))

"""
TESTING LOOP
"""

# Initialize new metrics
correct = 0
incorrect = 0
for x, y in zip(x_test, y_test):

    """
    Forward Pass Section
    """

    # Append bias term
    x = np.append(x, 1)

    # Instantiate hidden output
    hidden_nodes = np.copy(first_matrix)

    # Perform matrix multiplication
    for i1, value in enumerate(x):
        hidden_nodes[i1] = hidden_nodes[i1] * value

    # Get the sum for each output node
    hidden_nodes = np.sum(hidden_nodes, axis=0)

    # Activation Function
    hidden_nodes = activation(hidden_nodes)

    # Add bias term
    hidden_nodes = np.append(hidden_nodes, 1)

    # Instantiate final output
    output = np.zeros_like(second_matrix)
    for i2, value in enumerate(hidden_nodes):
        output[i2] = np.multiply(second_matrix[i2], value)

    # Get the sum for each output node
    output = np.sum(output, axis=0)

    # Activation function
    output = activation(output)

    # Get accuracy metrics
    if np.argmax(output) == np.argmax(y):
        correct += 1
    else:
        incorrect += 1

# Report metrics
print("Testing Accuracy: ", correct / (correct + incorrect))
