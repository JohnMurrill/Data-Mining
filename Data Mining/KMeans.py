import random
import matplotlib.pyplot as plt
import numpy

# Function to load in the datafile
def LoadData(filename):
        
        lines = open(filename, "rb")
        splitDataSet = []
        for i in lines:splitDataSet.append(str.split(i,','));
        for a in range(len(splitDataSet)):
            for j in range(len(splitDataSet[a])):
                if splitDataSet[a][j] != splitDataSet[a][-1]:
                    splitDataSet[a][j]=float(splitDataSet[a][j]);
        return splitDataSet[:-1]

# Main K-Means function, chooses 3 random centroids and clusters the data on them,
# and then iteratively recalculates the clusters and reclusters the data around the
# updataed centroids
def KMeans(data1, data2, k):
        
    centroids = [[], [], []]
    dataset = zip(data1,data2)
    for i in range(k):
        r= random.choice(range(i*50,(i+1)*50))
        centroids[i] = dataset[r]
    print centroids
    clusters = Cluster(centroids,dataset)

    for n in range(100):
        for i in range(len(centroids)):
            centroids[i] = UpdateCentroid(clusters[i])
        clusters = Cluster(centroids,dataset)       
    return clusters, [[centroids[0][0], centroids[1][0], centroids[2][0]], [centroids[0][1], centroids[1][1], centroids[2][1]]]

# Cluster function. Given centroids, it clusters the dataset based on the centroids
def Cluster(centroids, dataset):

    newClusters = [[],[],[]]
    dists = [0,0,0]
    for i in range(len(dataset)):
        for j in range(len(dists)):
            dists[j] = EuclidDist(dataset[i],centroids[j])
            r = dists.index(min(dists))
        newClusters[r].append(dataset[i])
    return newClusters

# Updates centroids to the mean of their cluster
def UpdateCentroid(data):
    x=0
    y=0
    for i in range(len(data)):
        x += data[i][0]
        y += data[i][1]
    meanX = float(x)/len(data)
    meanY = float(y)/len(data)
    return [round(meanX,4), round(meanY,4)]

# Euclidean Distance function to calculate the distance of a point from its
# centroid
def EuclidDist(vec1,vec2):
    vec1 = numpy.array(vec1)
    vec2 = numpy.array(vec2)
    return numpy.linalg.norm(vec1-vec2)

def PlotData(clusters, centroids, data1, data2):
    b=[[],[]]
    c=[[],[]]
    d=[[],[]]
    for i in range(len(clusters)):
         for j in range(len(clusters[i])):
             if i==0:
                 b[0].append(clusters[i][j][0])
                 b[1].append(clusters[i][j][1])
             elif i==1:
                 c[0].append(clusters[i][j][0])
                 c[1].append(clusters[i][j][1])
             else:
                 d[0].append(clusters[i][j][0])
                 d[1].append(clusters[i][j][1])
    print zip(centroids[0],centroids[1])
    print [len(b[0]),len(c[0]),len(d[0])]
    plt.subplot(211)
    plt.scatter(centroids[0],centroids[1],marker='o',color='black')
    plt.scatter(b[0],b[1],marker='.',color='red')
    plt.scatter(c[0],c[1],marker='.',color='blue')
    plt.scatter(d[0],d[1],marker='.',color='green')
    plt.subplot(212)
    plt.scatter(data1,data2,marker='.',color='black')
    plt.show()

def Main():
    filename ="iris.data"
    data = LoadData(filename)
    x=[0]*len(data)
    y=[0]*len(data)
    xsum=[0]*13
    ysum=[0]*13
    for i in range(len(data)):
        x[i]=data[i][0]
        y[i]=data[i][1]
    a,z=KMeans(x,y,3)
    PlotData(a,z,x,y)
    for i in range(len(data)):
        x[i]=data[i][2]
        y[i]=data[i][3]
    a,z=KMeans(x,y,3)
    PlotData(a,z,x,y)
    for i in range(len(data)):
        x[i]=data[i][2]
        y[i]=data[i][0]
    a,z=KMeans(x,y,3)
    PlotData(a,z,x,y)
Main()

