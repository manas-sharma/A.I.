# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:46:49 2019

@author: Manas
"""

import numpy as np
import matplotlib.pyplot as plt

def plotData(X,y,msg):
    pos = np.where(y==1)
    neg = np.where(y==0)
    plt.title(msg)
    plt.scatter(X[pos,0],X[pos,1],label='Accepted')
    plt.scatter(X[neg,0],X[neg,1],label='Rejected')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()

def mapFeature(x1,x2,degree):
    eqn = np.ones(n)[:,np.newaxis]
    for i in range (1,degree+1):
        for j in range (i+1):
            term = (x1**(i-j) * x2**j)[:,np.newaxis]
            eqn = np.hstack((eqn,term))
    return eqn

def mapFeatureContour(x1,x2,degree):
    eqn = np.ones(1)
    for i in range (1,degree+1):
        for j in range (i+1):
            term = (x1**(i-j) * x2**j)
            eqn = np.hstack((eqn,term))
    return eqn
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def computeCostReg(theta,X,y,lambda_):
    hx = sigmoid(np.matmul(X,theta))
    J = (-1/n) * np.sum(np.matmul(y.T,np.log(hx)) + np.matmul((1-y).T,np.log(1-hx))) + (lambda_/(2*n)) * np.sum(theta[1:]**2)
    return J

def gradientDescent(theta,X,y,lambda_,learningRate,maxIterations):
    for i in range(maxIterations):
        grad = gradient(theta,X,y,lambda_)
        theta = theta - learningRate*grad 
    return theta

def gradient(theta,X,y,lambda_):
    hx = sigmoid(np.matmul(X,theta))
    grad = (1/n) * np.matmul(X.T, hx - y) + (lambda_/n) * theta
    grad[0] = grad[0] - (lambda_/n) * theta[0]
    return grad

def accuracy(X, y, theta):
    y_hat = [sigmoid(np.matmul(X, theta)) >= 0.5]
    return np.mean(y_hat == y) * 100

data = np.loadtxt('./ex2/ex2data2.txt', dtype='float', delimiter=',')
plotData(data[:,:-1],data[:,-1],'Scatter plot of training data')
plt.show()

n = len(data)
X = data[:,:-1]
#X = np.hstack((np.ones((n,1)), X))
y = data[:,-1][:,np.newaxis]
degree = 6
lambda_ = 1
learningRate = 0.2
maxIterations = 1000

X = mapFeature(X[:,0],X[:,1],degree)

theta = np.zeros((X.shape[1],1))
print('Initial value of Cost Function=',computeCostReg(theta,X,y,lambda_))
theta = gradientDescent(theta,X,y,lambda_,learningRate,maxIterations)

# Decision Boundary
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i,j] = np.dot(mapFeatureContour(u[i], v[j], degree), theta)
        
plotData(data[:,:-1],data[:,-1],'Scatter plot of training data with decision boundary')
plt.contour(u,v,z.T,0)
plt.show()


print('Final value of Cost Function=',computeCostReg(theta,X,y,lambda_))
print('Model Accuracy=',accuracy(X, y, theta),'%')
