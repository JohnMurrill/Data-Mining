import random
from math import *
import csv
import copy
from pprint import pprint

def LoadData(filename):
        lines = open(filename, "rb")
        a=0
        splitDataSet = []
        for i in lines:splitDataSet.append(str.split(i,','));
        for i in range(len(splitDataSet)):
            splitDataSet[i][4]=splitDataSet[i][4].strip();
            if splitDataSet[i][0]== '\rSunny':
                splitDataSet[i][0] = 'Sunny'
            elif splitDataSet[i][0]== '\rRain':
                splitDataSet[i][0] = 'Rain'
            elif splitDataSet[i][0]== '\rOvercast':
                splitDataSet[i][0] = 'Overcast'
        return splitDataSet[1:]


def CountSplit(values,attribute,subList):
    goal = 'Yes'
    counter=[0]*len(values)
    for i in range(len(values)):
        yesCounter=0;
        totalCounter=0;
        for j in range(len(subList)):
            if subList[j][attribute]==values[i]:
                totalCounter+=1
                if subList[j][4]==goal:
                    yesCounter+=1
        counter[i]=[yesCounter,totalCounter-yesCounter]
    if attribute==4:
        return [[counter[0][0],counter[1][1]]]
    else:
        return counter
    
def Entropy(vals):
    a=[0]*len(vals)
    for i in range(len(vals)):
        for j in range(len(vals[i])):
            try:
                if vals[i][0] == 0:
                    a[i] = -vals[i][1]/float(sum(vals[i])) * log(vals[i][1]/float(sum(vals[i])),2)
                elif vals[i][1]==0:
                    a[i] = -vals[i][0]/float(sum(vals[i])) * log(vals[i][0]/float(sum(vals[i])),2)
                else:
                    a[i] = -vals[i][1]/float(sum(vals[i])) * log(vals[i][1]/float(sum(vals[i])),2) \
                           - vals[i][0]/float(sum(vals[i])) * log(vals[i][0]/float(sum(vals[i])),2)
            except ZeroDivisionError:
                a[i]=0
    return a

def FilterData(dataSet, attributes, ref):
    newData = []
    for i in range(len(attributes)):
        temp = []
        for j in range(len(dataSet)):
            if dataSet[j][ref]==attributes[i]:
                temp.append(dataSet[j])
        newData.append(temp)
    return newData

def InformationGain(totalEntropy, splitVals):
    a=0
    total=0
    for i in range(len(splitVals)):total+=sum(splitVals[i])
    if total != 0:
        for i in range(len(splitVals)):
            a += sum(splitVals[i])/float(total)*Entropy([splitVals[i]])[0]
        return totalEntropy - a
    else:
        return 0

def ID3Classifier(data, att, currentEntropy, goal):
    attributes = copy.deepcopy(att)
    atts = ['Outlook: ','Temp: ','Humidity: ','Wind: ']
    returnList=[]
    gains=[]
    bestAtt=0
    howTo = []
    for i in range(len(attributes)):gains.append(InformationGain(currentEntropy,CountSplit(attributes[i],i,data)))
    for i in range(len(gains)):
        if gains[i]==max(gains):
            bestAtt=i
    filtData = FilterData(data,attributes[bestAtt],bestAtt)
    attributes[bestAtt] = []
    for j in range(len(filtData)):
        if CheckPurity(filtData[j]):
            returnList.append(filtData[j])
            howTo.append(atts[bestAtt] + filtData[j][0][bestAtt])
            howTo.append(["Play: "+filtData[j][0][4]])
        else:
            howTo.append(atts[bestAtt] + filtData[j][0][bestAtt])
            howTo.append(ID3Classifier(filtData[j], attributes, currentEntropy-gains[bestAtt], goal))
            returnList.append(ID3Classifier(filtData[j], attributes, currentEntropy-gains[bestAtt],goal))
    if goal == "Classify":
        return howTo
    elif goal =="Tree":
        return returnList
            
def CheckPurity(data):
    pureTrue = True
    pureFalse = True
    for i in range(len(data)):
        if data[i][-1]=="Yes":
            pureFalse = False
        elif data[i][-1]=="No":
            pureTrue = False
    if pureTrue | pureFalse:
        return True
    else:
        return False

def print_list(lst, level=0):
    for l in lst[0:]:
        if type(l) is list:
            print_list(l, level + 1)
        else:
            print('    ' * level + '+---' + l)
    
def Main():
    data=[]
    filename = "tennis.csv"
    predValue=['Yes','No']
    initialEntropy=0
    data = LoadData(filename)
    attributes = [['Sunny','Overcast','Rain'],
    ['Hot','Cool','Mild'],
    ['High','Normal'],
    ['Weak','Strong']]
    gains=[]
    initialEntropy = Entropy(CountSplit(predValue,4,data))[0]
    gains = ID3Classifier(data,attributes,initialEntropy, "Tree")
    test = ID3Classifier(data,attributes,initialEntropy, "Classify")
    pprint(gains)
    print
    print_list(test)
Main()
