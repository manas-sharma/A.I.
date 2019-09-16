#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:38:00 2019

@author: manas
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

def displayData(X,y):
    fig,ax = plt.subplots(10,10,figsize=(8, 8))
    for i in range(10):
        for j in range (10):
            ax[i,j].axis('off')
            ax[i,j].imshow(X[np.random.randint(5000)].reshape((20,20),order='F'))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def prediction(theta1, theta2, X):
    a,_ = X.shape
    X = np.hstack((np.ones((a,1)), X))
    out_ih = sigmoid(np.matmul(X, theta1.T))
    out_ih = np.hstack((np.ones((a,1)), out_ih))
    out_ho = sigmoid(np.matmul(out_ih, theta2.T))
    return np.argmax(out_ho, axis = 1)+1
    

data = loadmat('./ex3/ex3data1.mat')
X = data['X']
y = data['y']
displayData(X,y)

data = loadmat('./ex3/ex3weights.mat')
theta1 = data['Theta1']
theta2 = data['Theta2']

input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10
predY =  prediction(theta1, theta2, X)
print(np.mean(predY == y.flatten()) * 100,'%')