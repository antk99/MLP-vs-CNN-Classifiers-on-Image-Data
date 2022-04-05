#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 21:58:43 2022

@author: lunadana

Each example is a 28x28 grayscale image
"""

import pandas as pd 
import numpy as np 
import mnist_reader

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

# Data normalization
X_train = X_train - np.mean(X_train, axis=0)
X_train = X_train/np.std(X_train, axis=0)
X_test = X_test - np.mean(X_test, axis=0)
X_test = X_test/np.std(X_test, axis=0)

# Get count of each class for data analysis
p, i = np.unique(y_train, return_counts=True)

# Data augmentation
flip_train = []

# Horizontally flipped images
counter = 0
for i in X_train:
    print(counter)
    i = i.reshape(28, 28)
    flip = np.fliplr(i)
    i = flip.reshape(1,784)
    flip_train = np.append(flip_train,i)

# New train dataset with the flipped images
flip_train = flip_train.reshape(60000,784)
    






