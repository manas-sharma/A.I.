# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 15:02:39 2019

@author: Manas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc

def plotData(X,y,msg):
    pos = np.where(y==1)
    neg = np.where(y==0)
    plt.title(msg)
    plt.scatter(X[pos,0],X[pos,1],label='Admitted')
    plt.scatter(X[neg,0],X[neg,1],label='Not Admitted')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def computeCost(theta,X,y):
    hx = sigmoid(np.matmul(X,theta))
    J = np.sum(-np.multiply(y,np.log(hx)) - np.multiply((1-y),np.log(1-hx)))/len(y)
    return J

def gradient(theta,X,y):
    return (1/n) * np.dot(sigmoid(np.matmul(X,theta)) - y, X)

def accuracy(X, y, theta, threshold):
    y_hat = sigmoid(np.matmul(X, theta)) >= threshold
    return np.mean(y_hat == y) * 100

data = np.loadtxt('./ex2/ex2data1.txt', dtype='float', delimiter=',')
plotData(data[:,:-1],data[:,-1],'Scatter plot of training data')
plt.show()

n = len(data)
X = data[:,:-1]
X = np.hstack((np.ones((n,1)), X))
y = data[:,-1]
y = y[:,np.newaxis]

_,b = X.shape
theta = np.zeros((b,1))
print('Initial value of Cost Function=',computeCost(theta,X,y))

# Finding optimal parameters(theta) using fmin_tnc
optParam = fmin_tnc(func=computeCost, x0=theta.flatten(), fprime=gradient, args=(X, y.flatten()))
theta = optParam[0][:,np.newaxis]


# Decision Boundary ---> woxo + w1x1 + w2x2 = 0
# x2(or y) = - (woxo + w1x1)/(w2)
x_range = np.linspace(np.min(X[:,1]), np.max(X[:,1]), num=10)
y_range = -1/theta[2][0]*(theta[0][0] + np.dot(theta[1][0],x_range))

plotData(data[:,:-1],data[:,-1],'Scatter plot of training data with decision boundary')
plt.plot(x_range, y_range)
plt.show()

print('Final value of Cost Function=',computeCost(theta,X,y))
print('Model Accuracy=',accuracy(X, y, theta, 0.5),'%')