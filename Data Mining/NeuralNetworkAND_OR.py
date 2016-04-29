from math import *
import random
import operator
from numpy import *

#Target function outputs.
AND    = [0,0,0,1]
OR     = [0,1,1,1]

#Learning rate and potential input cases for each perceptron
gamma = .5
test = [[1,0,0],[1,1,0],[1,0,1],[1,1,1]]

#Perceptron class to allow for multiple declerations later.
class Perceptron():
    LR = 1
    inputVector = [[1,0,0],[1,1,0],[1,0,1],[1,1,1]]
    weightVector = [0,0,0]
    target = []
    iterations =0
    def __init__(self, goalFunction):
        self.target = goalFunction
        for i in range(len(self.weightVector)):
            self.weightVector[i] = random.uniform(-5,5)

    #Evaluation function. Takes the dot product of the input vector and the weights
    def F(self,inp):
        return sum(map(operator.mul, inp, self.weightVector))

    #Training function. Iteratively adjusts the weights of the perceptron until it
    #matches the desired output. Note, since this is a single perceptron I used the
    #Delta Rule for adjusting weights rather than the full backpropagation rule for
    #multi-layer networks.
    #https://en.wikipedia.org/wiki/Delta_rule
    def Train(self):
        currentOutput = [0,0,0,0]
        while(True):
            for k in range(len(currentOutput)):
                currentOutput[k] = self.Output(self.inputVector[k])
            if currentOutput == self.target:
                break
            else:
                self.iterations += 1
                for i in range(len(self.inputVector)):
                    for j in range(len(self.inputVector[i])):
                        delta = -1*self.LR*(self.target[i]-self.Trigger(self.F(self.inputVector[i])))\
                                *self.inputVector[i][j]*\
                                (self.Trigger(self.F(self.inputVector[i]))*\
                                 (1-self.Trigger(self.F(self.inputVector[i]))))
                        self.weightVector[j] -= delta

    #Trigger function, evaluates for a Sigmoid
    def Trigger(self, O):
        return 1/(1+exp(-1*O))
    
    #Output function, returns 1 when the sigmoid is > .5 and 0 otherwise.
    def Output(self, I):
        O = self.F(I)   
        if self.Trigger(O) > .5:
            return 1
        else:
            return 0
        
#Examples / Output                
print 'AND Function:'
myP = Perceptron(AND)
print 'Initial weights: ',myP.weightVector
myP.Train()
print 'Final weights: ',myP.weightVector
print 'Took',myP.iterations,'learning iterations'
for i in range(len(test)):
    print myP.Output(test[i])

print
print 'OR Function'
twoP = Perceptron(OR)
print 'Initial weights: ',twoP.weightVector
twoP.Train()
print 'Final weights: ',twoP.weightVector
print 'Took ',twoP.iterations,'learning iterations'
for i in range(len(test)):
    print twoP.Output(test[i])
