# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 09:20:29 2017

@author: Manas
"""

import numpy as np

input_nodes = 2
hidden_nodes = 2
output_nodes = 1
learning_rate = 0.5
maxIterations = 15000
        
def sigmoidActivation(x):
    return 1 / (1 + np.exp(-x))
   
if __name__=="__main__":             
    
    i_To_hWeights=np.random.random(size=(hidden_nodes,input_nodes+1))
    h_To_oWeights=np.random.random(size=(output_nodes,hidden_nodes+1))
        
    i_To_hDelta = np.zeros((hidden_nodes), dtype=float)
    h_To_oDelta = np.zeros((output_nodes), dtype=float)
        
    hActivation = np.zeros((hidden_nodes,1), dtype=float)
    oActivation = np.zeros((output_nodes,1), dtype=float)
         
    iOutput = np.zeros((input_nodes+1,1), dtype=float)      
    hOutput = np.zeros((hidden_nodes+1,1), dtype=float)  
    oOutput = np.zeros((output_nodes), dtype=float)
    
    #For XOR
    trainingData_inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    trainingData_outputs = np.array([[0.0],[1.0],[1.0],[0.0]])
    
    print("Initial Random I to H weights:")
    print(i_To_hWeights)
    print("\nInitial Random H to O weights:")
    print(h_To_oWeights)
    
    for i in range(maxIterations):
        
        iOutput[:-1,0] = trainingData_inputs[i%4]
        iOutput[-1:,0] = 1.0
        
        hActivation = np.dot(i_To_hWeights,iOutput)
        hOutput[:-1,:] = sigmoidActivation(hActivation)
        hOutput[-1:,:] = 1.0
        
        oActivation = np.dot(h_To_oWeights,hOutput)
        oOutput = sigmoidActivation(oActivation)
        
        error = oOutput - trainingData_outputs[i%4]
        h_To_oDelta = sigmoidActivation(oActivation) * (1-sigmoidActivation(oActivation)) * error
        i_To_hDelta = sigmoidActivation(hActivation) * (1-sigmoidActivation(hActivation)) * np.dot(np.transpose(h_To_oWeights[:,:-1]), h_To_oDelta)
        
        i_To_hWeights -= learning_rate * np.dot(i_To_hDelta, np.transpose(iOutput))
        h_To_oWeights -= learning_rate * np.dot(h_To_oDelta, np.transpose(hOutput))
        
        if(i==maxIterations-4):
            print("\nTrained I to H weights:")
            print(i_To_hWeights)
            print("\nTrained H to O weights:")
            print(h_To_oWeights)
            
        if(i>=maxIterations-4):
            print("\nInput Pattern:")
            print(trainingData_inputs[i%4])
            print("Network Output: %f" %(oOutput))