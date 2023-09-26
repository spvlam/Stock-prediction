import numpy as np

class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer, metrics):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def train(self, X_train, y_train, epochs, batch_size):
        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                outputs = self.forward(X_batch)
                self.backward(X_batch, y_batch, outputs)
        
            loss = self.compute_loss(X_train, y_train)
            print(f"Epoch: {epoch + 1}, Loss: {loss}")

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def backward(self, inputs, targets, outputs):
        grad = self.loss.gradient(targets, outputs)
        for layer in reversed(self.layers):
            grad = layer.backward(inputs, grad)

    def compute_loss(self, inputs, targets):
        outputs = self.forward(inputs)
        return self.loss.compute(targets, outputs)

    def predict(self, inputs):
        return self.forward(inputs)

class LSTM:
    def __init__(self, units, return_sequences=False):
        self.units = units
        self.return_sequences = return_sequences
        self.weights = None
        self.bias = None
        self.activation = None
        self.prev_state = None
        self.prev_output = None

    def initialize(self, input_shape):
        self.n_x, n_h = input_shape[1], self.units

        self.weights = np.random.randn(n_h, self.n_x + n_h) * 0.01
        self.bias = np.zeros((n_h, 1))

    def forward(self, inputs):
        T = inputs.shape[0]
        n_h = self.units

        self.h = np.zeros((T + 1, n_h))
        self.h[-1] = np.zeros((n_h,))

        for t in range(T):
            x = inputs[t].reshape(-1, 1)

            self.h[t] = self.activation(np.dot(self.weights, np.vstack((x, self.h[t-1]))) + self.bias)

        if self.return_sequences:
            return self.h[:-1]
        else:
            return self.h[-1]

    def backward(self, inputs, grad_output):
        T = inputs.shape[0]
        n_h = self.units

        dW = np.zeros_like(self.weights)
        db = np.zeros_like(self.bias)
        dh_next = np.zeros_like(self.h[0])
        grad_input = np.zeros_like(inputs)

        for t in reversed(range(T)):
            x = inputs[t].reshape(-1, 1)
            dh = grad_output[t].reshape(-1, 1) + dh_next

            dW += np.dot(dh * self.activation.gradient(self.weights @ np.vstack((x, self.h[t]))), np.vstack((x, self.h[t])).T)
            db += dh * self.activation.gradient(self.bias)
            dh_next = dh * self.activation.gradient(self.weights @ np.vstack((x, self.h[t])))[n_h:]

            grad_input[t] = self.weights[:, :self.n_x].T @ (dh * self.activation.gradient(self.weights @ np.vstack((x, self.h[t]))))[:self.n_x]

        return grad_input

class Dense:
    def __init__(self, units):
        self.units = units
        self.weights = None
        self.bias = None
        self.activation = None

    def initialize(self, input_shape):
        self.n_x, n_h = input_shape[1], self.units

        self.weights = np.random.randn(n_h, self.n_x) * 0.01
        self.bias = np.zeros((n_h, 1))

    def forward(self, inputs):
        return self.activation(np.dot(self.weights, inputs.T) + self.bias)

    def backward(self, inputs, grad_output):
        dW = np.dot(grad_output, inputs)
        db = np.sum(grad_output, axis=1, keepdims=True)
        grad_input = np.dot(self.weights.T, grad_output)

        self.weights -= self.optimizer.learning_rate * dW
        self.bias -= self.optimizer.learning_rate * db

        return grad_input

class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def forward(self, inputs):
        self.mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        return inputs * self.mask

    def backward(self, inputs, grad_output):
        return grad_output * self.mask

class Activation:
    def __init__(self, activation_func):
        self.activation_func = activation_func

    def __call__(self, x):
        if self.activation_func == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_func == 'tanh':
            return self.tanh(x)
        elif self.activation_func == 'relu':
            return self.relu(x)
        else:
            raise ValueError(f"Invalid activation function: {self.activation_func}")

    def gradient(self, x):
        if self.activation_func == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation_func == 'tanh':
            return self.tanh_derivative(x)
        elif self.activation_func == 'relu':
            return self.relu_derivative(x)
        else:
            raise ValueError(f"Invalid activation function: {self.activation_func}")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
