# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:58:16 2017

@author: Manas
"""

from random import *
POP_SIZE = 20

def getFitness(binString):
    return sum(binString)

def crossover(binString1, binString2):
    randomIndex = randint(0,7)
    x1 = mutate(binString1[:randomIndex] + binString2[randomIndex:])
    x2 = mutate(binString2[:randomIndex] + binString1[randomIndex:])
    return x1,x2

def mutate(binString):
    binString[randint(0,7)] = randint(0,1)
    return binString

def naturalSelection(population):
    temp = [(getFitness(binString), binString) for binString in population]
    temp.sort(reverse = True)
    return [temp[i][1] for i in range(int(POP_SIZE/2))]

if __name__ == '__main__':

    totalFitness = 0
    population = [[randint(0,1) for _ in range(8)] for _ in range(POP_SIZE)]
    generation = 1
    fittestPopulation = naturalSelection(population)
    
    print("Initial Population:")
    for i in range(POP_SIZE):
            print(population[i])
            
    for i in range(int(POP_SIZE/2)):
        totalFitness += getFitness(fittestPopulation[i])

    while totalFitness < 8*int(POP_SIZE/2):
        offspring = []
        totalFitness = 0
        
        for _ in range(int(POP_SIZE/2)):
            x1,x2 = crossover(choice(fittestPopulation),choice(fittestPopulation))
            offspring.append(x1)
            offspring.append(x2)
            
        fittestPopulation = naturalSelection(offspring)
        print("\nGeneration %i:" % (generation))
        for i in range(int(POP_SIZE/2)):
            print(fittestPopulation[i])
        
        generation += 1
        for i in range(int(POP_SIZE/2)):
            totalFitness += getFitness(fittestPopulation[i])
