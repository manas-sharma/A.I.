# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:19:19 2019

@author: Manas
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

def computeCost(theta,X,y,lambda_):
    hx = sigmoid(np.matmul(X,theta))
    n = len(y)
    J = -1/n * np.sum(np.multiply(y,np.log(hx)) + np.multiply((1-y),np.log(1-hx))) + (lambda_/2*n) * np.sum(theta[1:]**2)
    return J

def gradientDescent(theta,X,y,lambda_,learningRate,maxIterations):    
    for i in range(maxIterations):
        grad = gradient(theta,X,y,lambda_)
        theta = theta - learningRate*grad
    return theta

def gradient(theta,X,y,lambda_):
    n = len(y)
    hx = sigmoid(np.matmul(X,theta))
    grad = (1/n) * np.matmul(X.T, hx - y) + (lambda_/n) * theta
    grad[0] = grad[0] - (lambda_/n) * theta[0]
    return grad

data = loadmat('./ex3/ex3data1.mat')
X = data['X']
y = data['y']
displayData(X,y)

k = 10
a,b = X.shape
X = np.hstack((np.ones((a,1)), X))
theta_arr = np.zeros((k,b+1))
lambda_ = 0.1
learningRate = 0.2
maxIterations = 2000

for i in range(k):
    if(i==0):
        digit = 10
    else:
        digit = i
    temp_theta = np.zeros((b+1,1))
    theta_arr[i] = (gradientDescent(temp_theta,X,y == digit,lambda_,learningRate,maxIterations).T)
    
predIndex = np.argmax(np.matmul(X,theta_arr.T), axis = 1)
predY = [10 if (i==0) else i for i in predIndex]
print(np.mean(predY == y.flatten()) * 100,'%')