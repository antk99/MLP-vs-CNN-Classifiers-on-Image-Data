# -*- coding: utf-8 -*-

import numpy as np
import warnings
import pandas as pd
import random


warnings.filterwarnings('ignore')


def ReLu(arr):
    return np.maximum(arr, 0)

def leakyReLu(arr, gamma=0.1):
    return np.maximum(arr, 0) - gamma*np.minimum(0, arr)


logistic = lambda z: 1. / (1 + np.exp(-z))
    
    
def softmax(arr):
    e = np.exp(arr)
    return e/e.sum()

def tanh(arr):
    return 2*logistic(arr) - 1


def evaluate_acc(y_test, y_pred):
    """
    Evaluates the accuracy of a model's prediction
    :param y_test: np.ndarray - the true labels
    :param y_pred: np.ndarray - the predicted labels
    :return: float - prediction accuracy
    """
    #print(y_pred)
    y_test = y_test.reshape(1, y_test.shape[0])
    #print(y_test)
    return np.sum(y_pred == y_test) / len(y_pred)

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


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
        y = one_hot(y, self.num_outputs)
        
        def gradient0(x, y, params):
            w = params # D x C
            yh = softmax(np.dot(x, w))  # N x C
            dy = yh - y  # N x C
            dw = np.dot(x.T, dy) / N  # D x C
            return [dw]
        
        def gradient1(x, y, params):
            #print("INSIDE GRADIENT FUNCTION")
            
            '''
            TO GET ACCURACY AT EVERY ITERATION
            
            global first
            
            if False: #not first: 
                global new_y_test
                global new_X_test
                global list_of_acc
                
                y_pred = np.argmax(self.predict(new_X_test), axis=1)
                x = evaluate_acc(new_y_test, y_pred)
                print(x)
                list_of_acc.append(x)
                
            
            first = False
            '''
            
            v, w = params
            q = np.dot(x, v)
            q_01 = np.where(q > 0, 1, 0)
            z = self.activation_fn(q)  # N x M
            yh = softmax(np.dot(z, w))  # N x C
            dy = yh - y  # N x C
            dw = np.dot(z.T, dy) / N  # M x C
            dz = np.dot(dy, w.T)    # N x M
            #print("x: ")
            #print(x)
            #print("q_01: ")
            #print(q_01)
            #print("dz: ")
            #print(dz)
            #print("dz * q_01: ")
            #print(dz * q_01)
            #print("dv before dividing by N: ")
            #print(np.dot(x.T, dz * q_01))
            dv = np.dot(x.T, dz * q_01)/N   # D x M
            
            ################### leaky relu dv:
            #gamma = 0.1
            #q_gamma1 = np.where(q > 0, 1, gamma)
            #dv = np.dot(x.T, dz * q_gamma1)/N     
            
            ################### tanh dv:
            #q_tanh = 1-tanh(q)**2
            #dv = np.dot(x.T, dz * q_tanh)/N     
                
            #print("dw: ")
            #print(dw)
            #print("dv: ")
            #print(dv)
            #print("OUUUUUUUTSIDE GRADIENT FUNCTION")
            
            return [dv, dw]
        
        def gradient2(x, y, params): # assuming L hidden units for layer 1, then M for layer 2
            u, v, w = params # U: D x L -- V: L x M -- W: M x C 
            print(u.shape)
            print(v.shape)
            print(w.shape)
            print(x.shape)
            print(y.shape)
            r = np.dot(x, u) # N x L
            r_01 = np.where(r > 0, 1, 0) # N x L
            s = self.activation_fn(r) # N x L
            q = np.dot(s, v) # N x M
            q_01 = np.where(q > 0, 1, 0)
            z = self.activation_fn(q)  # N x M
            yh = softmax(np.dot(z, w))  # N x C
            dy = yh - y  # N x C
            dw = np.dot(z.T, dy) / N  # M x C
            dz = np.dot(dy, w.T)    # N x M
            
            
            ################### leaky relu dv:
            #gamma = 0.1
            #q_gamma1 = np.where(q > 0, 1, gamma)
            #dv = np.dot(x.T, dz * q_gamma1)/N    
            
            
            ################### tanh dv:
            #q_tanh = 1-tanh(q)**2
            #dv = np.dot(x.T, dz * q_tanh)/N    
            
            dv = np.dot(s.T, dz * q_01)/N   # D x M (x is now replaced by s at this stage compared to 1 layer)
            dq = np.dot(dz * q_01, v)
            du = np.dot(x.T, dq * r_01)/N
            
            
            ################### tanh du:
            #r_tanh = 1-tanh(x)**2
            #dq = np.dot(dz * q_tanh, v)
            #du = np.dot(x.T, dq * r_tanh)/N    
             
            ################### leaky relu du:
            #gamma = 0.1
            #r_gamma1 = np.where(x > 0, 1, gamma)
            #dq = np.dot(dz * q_gamma1, v)
            #du = np.dot(x.T, dq * r_gamma1)/N    
            
            return [du, dv, dw]

        print("\n gradient descent running \n")
        # for gradient1
        self.params = GradientDescent().run(gradient1, x, y, [self.weights[0], self.weights[1]])
        # for gradient2
        #self.params = GradientDescent().run(gradient2, x, y, [self.weights[0], self.weights[1], self.weights[2]])
        
        return self

    def predict(self, x):
        print("IN PREDICT")
        output = x
        for weight_matrix in self.params:
            #print(weight_matrix)
            #print(np.dot(output, weight_matrix))
            #print(np.dot(output, weight_matrix).shape)
            output = self.activation_fn(np.dot(output, weight_matrix))
        
        #v, w = self.params
        #z = self.activation_fn(np.dot(x, v))
        #yh = softmax(np.dot(z, w))
        print(output[0])
        print(softmax(output[0]))
        return softmax(output)


class GradientDescent:

    def __init__(self, learning_rate=.001, max_iters=1e4, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.epsilon = epsilon

    def run(self, gradient_fn, x, y, params):
        global temp
        
        norms = np.array([np.inf])
        t = 1
        while np.any(norms > self.epsilon) and t < self.max_iters:
            
            
            print(t)
            grad = gradient_fn(x, y, params)
            #print(grad[0].shape)
            if np.isnan(grad[0]).any():
                print("WE BREAK (nan DETECTED), HERE IS THE GRADIENT THAT CAUSED THE BREAK")
                # print(grad)
                break
            
            flatten_arr = np.ravel(params[0])
            #temp = np.sum((abs(flatten_arr) > 1e-8))
            #print(np.any(abs(flatten_arr) < 1e-15))
            if np.any(abs(flatten_arr) < 1e-15):
                #print("WE BREAK (VALUES TOO LOW), HERE IS THE GRADIENT THAT CAUSED THE BREAK")
                #print(grad)
                print("VALUE THAT CAUSED THE BREAK: ")
                print(min(abs(flatten_arr)))
                break

            #print(params[0].shape)
            for p in range(len(params)):
                #print(f"PARAMETER {p} BEFORE CHANGE: ")
                #print(params[p])
                #print()
                params[p] -= self.learning_rate * grad[p]
                #print(f"PARAMETER {p} AFTER CHANGE: ")
                #print(params[p])
                #print()
                #print(f"VALUE OF GRAD {p}: ")
                #print(grad[p])
                #print()
            t += 1
            norms = np.array([np.linalg.norm(g) for g in grad])
        return params


if __name__ == '__main__':
    import numpy as np
    import mnist_reader

    temp = []

    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # Data normalization
    X_train = X_train - np.mean(X_train)
    X_train = (X_train / np.std(X_train))
    X_test = X_test - np.mean(X_test)
    X_test = (X_test / np.std(X_test))


    

    list_of_acc = []

    first = True

    model = MLP(X_train.shape[1], 10, [128], activation_fn=ReLu)
    model.fit(X_train, y_train)
    #print(model.params)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(evaluate_acc(y_test, y_pred))

    





