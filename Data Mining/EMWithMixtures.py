import random
import numpy as np
from scipy.stats import multivariate_normal

# Function to load in the datafile
def LoadData(filename):
        lines = open(filename, "rb")
        splitDataSet = []
        for i in lines:splitDataSet.append(str.split(i,',')[:-1]);
        for a in range(len(splitDataSet)):
            for j in range(len(splitDataSet[a])):
                if splitDataSet[a][j] != splitDataSet[a][-1]:
                    splitDataSet[a][j]=float(splitDataSet[a][j]);
        return splitDataSet[:-1]

# This function updates the alpha weights of each distribution
def UpdateAlpha(Clusters): 
    weights = [[0],[0],[0]]
    N = len(Clusters)
    for i in range(len(Clusters[0])):
        total=0
        for j in range(len(Clusters)):
            total += Clusters[j][i]
        weights[i] = total/N
    return weights

# Function to update the Mu vector for the paramter set of the gaussian
def UpdateMuVector(data, weightMatrix):
    mu = [[0],[0],[0]]
    for i in range(len(data)):
        for j in range(len(weightMatrix[i])):
            mu[j] += np.multiply(data[i].tolist()[0],weightMatrix[i][j])
    for j in range(len(mu)):
        N = sum(np.array(weightMatrix).T[j])
        mu[j] = np.divide(mu[j],N)
    return mu

# Multivariate gaussian PDF function 
def GaussianDist(x, mu, sigma):
    return multivariate_normal.pdf(x,mu,sigma)

# mu is a vector of the mu vectors for each cluster
# This fuction returns the weighted covariance matrix of a parameter set.
def CovarianceMatrix(data, mu, weights):
    newSigs = []
    for i in range(len(mu)):
        newSigs.append([0])
        for j in range(len(data)):
            x = np.matrix([a_i - b_i for a_i, b_i in zip(data[j].tolist()[0], mu[i])])
            newSigs[i] += np.multiply(weights[j][i],np.dot(x.T,x))
        newSigs[i] = np.divide(newSigs[i],sum(np.array(weights).T.tolist()[i]))
    return newSigs

        
# Parameters should be in the form [alpha, mu, sigma]
# Function performs the E step of EM, finding the membership weights of each item
# relative to the total of all weights in all parameter sets.
def E_Step(data, parameters):
    membershipList = []
    for i in range(len(data)):
        membershipList.append([])
        for j in range(len(parameters)):
            membershipList[i].append(parameters[j][0]*GaussianDist(data[i], parameters[j][1], parameters[j][2]))
    for i in range(len(membershipList)):
        membershipList[i] = np.divide(membershipList[i],float(sum(membershipList[i]))).tolist()
    return membershipList

# M Step of EM, updates the parameters of each distribution
def M_Step(data, weightMatrix):
    parameters = []
    for i in range(len(weightMatrix[0])):
        parameters.append([])
        parameters[i].append(UpdateAlpha(weightMatrix)[i])
        x = UpdateMuVector(data, weightMatrix)
        parameters[i].append(UpdateMuVector(data, weightMatrix)[i].tolist())
        parameters[i].append(CovarianceMatrix(data, x ,weightMatrix)[i])
    return parameters

# Full EM Function call. Initializes uniform paramters with random choices of points
# as the initial Mu's. Covariance is initialized as the covariance of the dataset, 
# and alpha is set to 1/N. Function iteratively calls the E and M steps for a set
# number of epochs before terminating.
def EM(data, numClusts):
    basicCov = np.cov(data[random.choice(range(150))].tolist()[0])
    parameters = [[1/3.0,data[random.choice(range(0,50))].tolist()[0],basicCov],\
    [1/3.0,data[random.choice(range(51,100))].tolist()[0],basicCov],\
    [1/3.0,data[random.choice(range(101,150))].tolist()[0],basicCov]]
    

    for i in range(100):
        weightMatrix = E_Step(data, parameters)
        parameters = M_Step(data, weightMatrix)

    for i in range(len(parameters)):
        for j in range(len(parameters[i])):
            print parameters[i][j]
    
    for i in range(len(weightMatrix)):
        print [ '%.4f' % elem for elem in weightMatrix[i] ]

# Main function, loads the data and runs the program.
def Main():
    filename ="iris.data"
    numClusts = 3
    data = np.matrix(LoadData(filename))
    data = data.astype(np.float)
    EM(data, numClusts)
Main()

