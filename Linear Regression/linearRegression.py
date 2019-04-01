# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:10:49 2019

@author: Manas
"""

import numpy as np
import matplotlib.pyplot as plt

def plotData(X,y):
    plt.title('Single Variable Linear Regression')
    plt.scatter(X,y)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    #plt.show()

def computeCost(X,y,theta):
    hx = np.matmul(X,theta)
    J = np.sum(np.power(hx-y, 2))/(2*len(y))
    return J

def gradientDescent(X,y,theta,learningRate,maxIterations):
    for i in range(maxIterations):
        hx = np.matmul(X,theta)
        temp = hx - y
        theta = theta - (learningRate/n) * np.dot(X.T, temp)
    return theta

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mu)/std
    return X,mu,std

def oneVariable():
    X = np.reshape(data[:,0],(n,1))
    y = np.reshape(data[:,1],(n,1))
    plotData(X,y)
    
    X = np.append(np.ones((n,1)),X,axis=1)
    theta = np.zeros((2,1))
    maxIterations = 1500
    learningRate = 0.01
    print('Initial value of Cost Function(Single-Variable): ',computeCost(X, y, theta))
    
    theta = gradientDescent(X,y,theta,learningRate,maxIterations)
    print('Final value of Cost Function(Single-Variable): ',computeCost(X, y, theta))
        
    plt.plot(X[:,1],np.matmul(X,theta),c='magenta')
    plt.show()

def multiVariable():
    X = np.reshape(data[:,:-1],(n,2))
    y = np.reshape(data[:,-1],(n,1))
    X,mu,std = featureNormalize(X)
    
    X = np.append(np.ones((n,1)),X,axis=1)
    theta = np.zeros((3,1))
    maxIterations = 1500
    learningRate = 0.01
    print('Initial value of Cost Function(Multi-Variable): ',computeCost(X, y, theta))
    
    theta = gradientDescent(X,y,theta,learningRate,maxIterations)
    print('Final value of Cost Function(Multi-Variable): ',computeCost(X, y, theta))

#Single Variable Linear Regression
data = np.loadtxt('./ex1/ex1data1.txt', dtype='float', delimiter=',')
n = len(data)
oneVariable()

#Multi-Variable Linear Regression
data = np.loadtxt('./ex1/ex1data2.txt', dtype='float', delimiter=',')
n = len(data)
multiVariable()
