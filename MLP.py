import numpy as np
import warnings

warnings.filterwarnings('ignore')

logistic = lambda z: 1. / (1 + np.exp(-z))


class MLP:

    def __init__(self, M=64):
        self.M = M

    def fit(self, x, y, optimizer):
        N, D = x.shape

        def gradient(x, y, params):
            v, w = params
            z = logistic(np.dot(x, v))  # N x M
            yh = logistic(np.dot(z, w))  # N
            dy = yh - y  # N
            dw = np.dot(z.T, dy) / N  # M
            dz = np.outer(dy, w)  # N x M
            dv = np.dot(x.T, dz * z * (1 - z)) / N  # D x M
            dparams = [dv, dw]
            return dparams

        w = np.random.randn(self.M) * .01
        v = np.random.randn(D, self.M) * .01
        params0 = [v, w]
        self.params = optimizer.run(gradient, x, y, params0)
        return self

    def predict(self, x):
        v, w = self.params
        z = logistic(np.dot(x, v))  # N x M
        yh = logistic(np.dot(z, w))  # N
        return yh


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
