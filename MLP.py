import numpy as np
import warnings

warnings.filterwarnings('ignore')


def ReLu(arr):
    return np.maximum(arr, 0)


def softmax(arr):
    e = np.exp(arr)
    return e/e.sum()


def evaluate_acc(y_test, y_pred):
    """
    Evaluates the accuracy of a model's prediction
    :param y_test: np.ndarray - the true labels
    :param y_pred: np.ndarray - the predicted labels
    :return: float - prediction accuracy
    """
    return np.sum(y_pred == y_test) / y_pred.shape[0]


class MLP:

    def __init__(self, num_inputs, num_outputs, num_hidden_units=[64, 64], activation_fn=ReLu):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = len(num_hidden_units)
        self.size_layers = num_hidden_units
        self.activation_fn = activation_fn

        self.weights = []
        layers = [num_inputs] + num_hidden_units + [num_outputs]

        # initializing weights and biases (bias ignored for now)
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * .01)  # idk why * .01 it was in the slides

    def fit(self, x, y):
        N, D = x.shape

        def gradient1(x, y, params):
            v, w = params
            q = np.dot(x, v)
            q_01 = np.where(q > 0, 1, 0)
            z = ReLu(q)  # N x M
            yh = softmax(np.dot(z, w))  # N
            dy = yh - y  # N
            dw = np.dot(z.T, dy) / N  # M x C
            dz = np.dot(dy, w.T)    # N x M
            dv = np.dot(x.T, dz * q_01)/N   # D x M
            return [dw, dv]

        self.params = GradientDescent().run(gradient1, x, y, [self.weights[0], self.weights[1]])
        return self

    def predict(self, x):
        output = x
        for weight_matrix in self.weights:
            output = self.activation_fn(np.dot(output, weight_matrix))
        return output


class GradientDescent:

    def __init__(self, learning_rate=.001, max_iters=1e4, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.epsilon = epsilon

    def run(self, gradient_fn, x, y, params):
        norms = np.array([np.inf])
        t = 1
        while np.any(norms > self.epsilon) and t < self.max_iters:
            grad = gradient_fn(x, y, params)
            for p in range(len(params)):
                params[p] -= self.learning_rate * grad[p]
            t += 1
            norms = np.array([np.linalg.norm(g) for g in grad])
        return params


if __name__ == '__main__':
    pass