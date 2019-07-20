# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 18:55:31 2017

@author: Manas
"""

from numpy import *
from matplotlib import pyplot as plt
from matplotlib import patches as patches

def getBMU(m,n,nodeWeights,currentPattern):
    minDist = row = col = 0
    for i in range(m):
        for j in range(n):
            currentNode = nodeWeights[i][j]
            eucliDist = sum((currentPattern-currentNode)**2)**0.5
            if(i==0 and j==0):
                minDist = eucliDist
                
            if(minDist > eucliDist):
                minDist = eucliDist
                row = i
                col = j
                
    return (row,col)

def plotColors(m,n,nodeWeights,msg):
    fig = plt.figure()
    
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((0, m+1))
    ax.set_ylim((0, n+1))
    
    for x in range(1, m + 1):
        for y in range(1, n + 1):
            rgb = nodeWeights[x-1][y-1]
            ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1, facecolor=rgb))
    plt.title("Colors "+msg+" training")
    plt.show()

if __name__ == "__main__":
    
    m = n = 8
    initLearningRate = 2.0
    maxIterations = 8000
    patterns = random.random((100,3))
    initRadius = max(m,n)/2
    lambdaConstant = maxIterations / log(initRadius)
    nodeWeights = random.random((m,n,3))
    plotColors(m,n,nodeWeights,"before")
    
    for i in range(maxIterations):

        randomIndex = random.randint(0,100)
        currentPattern = patterns[randomIndex]
        
        row,col = getBMU(m,n,nodeWeights,currentPattern)        
        learningRate = initLearningRate * exp(-i/lambdaConstant)
        newRadius = initRadius * exp(-i/lambdaConstant)
            
        for x in range(m):
            for y in range(n):
                dist = ((row-x)**2 + (col-y)**2)**0.5
                if(dist<=newRadius):
                    theta = exp(-(dist**2)/(2 * learningRate**2))
                    nodeWeights[x][y] = nodeWeights[x][y] + (theta*learningRate*(currentPattern-nodeWeights[x][y]))
    
    plotColors(m,n,nodeWeights,"after")


   