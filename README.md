# Neural Network
Deep learning employs deep neural networks, which are artificial neural networks with multiple layers that mimic the 
neural networks in a human brain. These networks are designed to learn and extract increasingly complex, abstract 
representations of data as information flows through each layer.
## How Does it Work?
Deep learning works by training artificial neural networks with multiple layers, allowing them to learn hierarchical 
representations of data and make predictions or generate outputs. 
## Neural Network Architectures
Neural network architectures are the **building blocks** of deep learning models. They consist of interconnected nodes, 
called neurons, which are organized in layers. Each neuron receives inputs, computes mathematical operations, 
and produces outputs.
## Main Components of Neural Network Architecture
The main components of a neural network architecture are:
- **Input Layer:** The input layer is the initial layer of the neural network and is responsible for receiving 
the input data. Each neuron in the input layer represents a feature or attribute of the input data.
- **Hidden Layers:** Hidden layers are the intermediate layers between the input and output layers. They perform 
computations and transform the input data through a series of weighted connections. The number of hidden layers and 
the number of neurons in each layer can vary depending on the complexity of the task and the amount of data available.
- **Neurons (Nodes):** Neurons, also known as nodes, are the individual computing units within a neural network.
Each neuron receives input from the previous layer or directly from the input layer, performs a computation using 
weights and biases, and produces an output value using an activation function.
- **Weights and Biases:** Weights and biases are parameters associated with the connections between neurons. 
The weights determine the strength or importance of the connections, while the biases introduce a constant that helps 
control the neuron's activation. These parameters are adjusted during the training process to optimize the network's 
performance.
- **Activation Functions:** Activation functions are special mathematical formulas that add non-linear behavior 
to the network and allow it to learn complex patterns. Common activation functions include the _Sigmoid Function_, 
the _Rectified Linear Unit (ReLU)_, and the _Hyperbolic Tangent (tanh) Function_. Each neuron applies the activation 
function to the weighted sum of its inputs to produce the output. Each function behaves differently and has its 
own characteristics. They help the network process and transform the input information, making it more suitable 
for capturing the complexity of real-world data. Activation functions help neurons make decisions and capture 
intricate relationships in the data, making neural networks powerful tools for pattern recognition and accurate 
predictions.
- **Output Layer:** The output layer is the final layer of the neural network that produces the network's 
predictions or outputs after processing the input data. The number of neurons in the output layer depends 
on the nature of the task. For _binary classification tasks_, where the goal is to determine whether something 
belongs to one of two categories (e.g., yes/no, true/false), the output layer typically consists of a _single 
neuron_. For _multi-class classification_ tasks, where there are more than two categories to consider (e.g., classifying 
images into different objects), the output layer consists of _multiple neurons_. 
- **Loss Function:** The loss function measures the discrepancy between the network's predicted output and 
the true output. It quantifies the network's performance during training and serves as a guide for adjusting 
the weights and biases. For example, if the task involves predicting _numerical values_, like estimating 
the price of a house based on its features, the _mean squared error loss function_ may be used. This function 
calculates the average of the squared differences between the network's predicted values and the true values. 
On the other hand, if the task involves _classification_, where the goal is to assign input data to different categories, 
a loss function called _cross-entropy_ is often used. Cross-entropy measures the difference between the predicted 
probabilities assigned by the network and the true labels of the data. It helps the network understand _how well 
it is classifying_ the input into the correct categories.\
These components work together to process input data, propagate information through the network, and produce the 
desired output. The weights and biases are adjusted during the training process through optimization algorithms 
to minimize the loss function and improve the network's performance.
## Loss Functions
Key Loss Functions are:
- **Mean Squared Error (MSE) Loss Function** is the sum of squared differences between the entries in the prediction 
vector y and the ground truth vector y_hat. 
![MSE loss function](./img/1_loss%20functions.png)
- You divide the sum of squared differences by N, which corresponds to the 
length of the vectors. If the output y of your neural network is a vector with multiple entries then N is the number 
of the vector entries with y_i being one particular entry in the output vector. The mean squared error loss function 
is the perfect loss function if you're dealing with a regression problem. That is, if you want your neural network 
to predict a continuous scalar value.
- **Cross-Entropy Loss Function** - a loss function that measure the error between a predicted probability and 
the label which represents the actual class. The output of a neural network must be in a range between zero and one.
![MSE loss function](./img/6_loss%20functions.png)
The label vector y_hat is one hot encoded which means the values in this vector can only take discrete values of either 
zero or one. The entries in this vector represent different classes. The values of these entries are zero, except for 
a single entry which is one. This entry tells us the class into which we want to classify the input feature vector x.
- **Mean Absolute Percentage Error (MAPE)** function measures the performance of a neural network during demand 
forecasting tasks - the area of predictive analytics dedicated to predicting the expected demand for a good or service 
in the near future. For example in retail, we can use demand forecasting models to determine the amount of 
a particular product that should be available and at what price. It is also known as mean absolute percentage 
deviation (MAPD).
![Mean Absolute Percentage Error](./img/8_loss%20functions.png)
## Choosing Activation Functions
For Hidden Layers Typically use:
- ReLU, 
- Leaky ReLU, 
- Swish.\
For Output Layer:
- Binary Classification: Sigmoid
- Multi-Class Classification: Softmax
- Regression: Linear
## Optimization Algorithms For Training Neural Network
The right optimization algorithm can reduce training time exponentially. Optimizers are algorithms or methods used 
to change the attributes of your neural network such as weights and learning rate in order to reduce the losses:
- **Gradient Descent** is the most basic but most used optimization algorithm. It’s used heavily in linear regression 
and classification algorithms, backpropagation in neural networks. It is a first-order optimization algorithm which 
is dependent on the first order derivative of a loss function. It calculates that which way the weights should 
be altered so that the function can reach a minima. Through backpropagation, the loss is transferred from one layer 
to another and the model’s parameters also known as weights are modified depending on the losses so that the loss 
can be minimized. Algorithm: θ=θ−α⋅∇J(θ).
- **Stochastic Gradient Descent** is a variant of Gradient Descent. It tries to update the model’s parameters more 
frequently. In this, the model parameters are altered after computation of loss on each training example. 
So, if the dataset contains 1000 rows SGD will update the model parameters 1000 times in one cycle of dataset 
instead of one time as in Gradient Descent.
- **Mini-Batch Gradient Descent** is best among all the variations of gradient descent algorithms. It is 
an improvement on both SGD and standard gradient descent. It updates the model parameters after every batch. 
So, the dataset is divided into various batches and after every batch, the parameters are updated.
- **Momentum** was invented for reducing high variance in SGD and softens the convergence. It accelerates 
the convergence towards the relevant direction and reduces the fluctuation to the irrelevant direction. 
One more hyperparameter is used in this method known as momentum symbolized by ‘γ’.
- **Nesterov Accelerated Gradient** calculates the cost based on this future parameter rather than the current one.
- **Adagrad** changes the learning rate ‘η’ for each parameter and at every time step ‘t’. It’s a type second order 
optimization algorithm. It works on the derivative of an error function.
- **AdaDelta** is an extension of AdaGrad which tends to remove the decaying learning Rate problem of it. Instead of 
accumulating all previously squared gradients, Adadelta limits the window of accumulated past gradients to some 
fixed size w. In this exponentially moving average is used rather than the sum of all the gradients.
- **Adam (Adaptive Moment Estimation)** works with momentum's of first and second order. The intuition behind the Adam 
is that we don’t want to roll so fast just because we can jump over the minimum, we want to decrease the velocity 
a little bit for a careful search. In addition to storing an exponentially decaying average of past squared gradients 
like AdaDelta, Adam also keeps an exponentially decaying average of past gradients M(t).
## Resources
[Deep learning](https://www.functionize.com/blog/neural-network-architectures-and-generative-models-part1)\
[Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)\
[How Loss Functions Work in Neural Networks](https://builtin.com/machine-learning/loss-functions)\
[MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)\
[Various Optimization Algorithms For Training Neural Network](https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6)\

