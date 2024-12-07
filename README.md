# Neural Network
Deep learning employs deep neural networks, which are artificial neural networks with multiple layers that mimic the 
neural networks in a human brain. These networks are designed to learn and extract increasingly complex, abstract 
representations of data as information flows through each layer.
## How Does it Work?
Deep learning works by training artificial neural networks with multiple layers, allowing them to learn hierarchical 
representations of data and make predictions or generate outputs. 
## Neural Network Architectures
Neural network architectures are the building blocks of deep learning models. They consist of interconnected nodes, 
called neurons, which are organized in layers. Each neuron receives inputs, computes mathematical operations, 
and produces outputs.
## Main Components of Neural Network Architecture
Neural network architectures consist of several components that work together to process and learn from data. The main 
components of a neural network architecture are:
- **Input Layer:** The input layer is the initial layer of the neural network and is responsible for receiving 
the input data. Each neuron in the input layer represents a feature or attribute of the input data.
- **Hidden Layers:** Hidden layers are the intermediate layers between the input and output layers. 
They perform computations and transform the input data through a series of weighted connections. The number 
of hidden layers and the number of neurons in each layer can vary depending on the complexity of the task and 
the amount of data available.
- **Neurons (Nodes):** Neurons, also known as nodes, are the individual computing units within a neural network.
Each neuron receives input from the previous layer or directly from the input layer, performs a computation using 
weights and biases, and produces an output value using an activation function.
- **Weights and Biases:** Weights and biases are parameters associated with the connections between neurons. 
The weights determine the strength or importance of the connections, while the biases introduce a constant that helps 
control the neuron's activation. These parameters are adjusted during the training process to optimize the network's 
performance.
- **Activation Functions:** Activation functions are special mathematical formulas that add non-linear behavior 
to the network and allow it to learn complex patterns. Common activation functions include the sigmoid function, 
the rectified linear unit (ReLU), and the hyperbolic tangent (tanh) function. Each neuron applies the activation 
function to the weighted sum of its inputs to produce the output. Each function behaves differently and has its 
own characteristics. They help the network process and transform the input information, making it more suitable 
for capturing the complexity of real-world data. Activation functions help neurons make decisions and capture 
intricate relationships in the data, making neural networks powerful tools for pattern recognition and accurate 
predictions.
- **Output Layer:** The output layer is the final layer of the neural network that produces the network's 
predictions or outputs after processing the input data. The number of neurons in the output layer depends 
on the nature of the task. For binary classification tasks, where the goal is to determine whether something 
belongs to one of two categories (e.g., yes/no, true/false), the output layer typically consists of a single 
neuron. For multi-class classification tasks, where there are more than two categories to consider (e.g., classifying 
images into different objects), the output layer consists of multiple neurons. 
- **Loss Function:** The loss function measures the discrepancy between the network's predicted output and 
the true output. It quantifies the network's performance during training and serves as a guide for adjusting 
the weights and biases. For example, if the task involves predicting numerical values, like estimating 
the price of a house based on its features, the mean squared error loss function may be used. This function 
calculates the average of the squared differences between the network's predicted values and the true values. 
On the other hand, if the task involves classification, where the goal is to assign input data to different categories, 
a loss function called cross-entropy is often used. Cross-entropy measures the difference between the predicted 
probabilities assigned by the network and the true labels of the data. It helps the network understand how well 
it is classifying the input into the correct categories.
These components work together to process input data, propagate information through the network, and produce the 
desired output. The weights and biases are adjusted during the training process through optimization algorithms 
to minimize the loss function and improve the network's performance.
## Choosing Activation Functions
For Hidden Layers Typically use:
- ReLU, 
- Leaky ReLU, 
- Swish
For Output Layer:
- Binary Classification: Sigmoid
- Multi-Class Classification: Softmax
- Regression: Linear
## Resources
[Deep learning](https://www.functionize.com/blog/neural-network-architectures-and-generative-models-part1)\
[Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)\
[MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
