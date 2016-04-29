from random import *
from math import *
import csv
import itertools



# This function takes in a filename as a string, opens the file, and returns the given data file as a matrix of integers.
# Each line is returned as an element.
def LoadWordData(filename):
        lines = open(filename, "rb")
        a=0
        dataset = []
        for i in lines:dataset.append(str.split(i));
        for a in range(len(dataset)):
                for j in range(len(dataset[a])):
                        dataset[a][j]=int(dataset[a][j]);
        return dataset

# Randomizes the data and answer files into a symmetric random order so that when the model begins to process the data,
# The data is randomized and not ordered.
def ShuffleFile(dataFile, answerFile):
        order = range(len(dataFile))
        shuffle(order)
        newDataOrder = [0]*len(dataFile)
        newAnswOrder = [0]*len(dataFile)
        for i in range(len(order)):
                newDataOrder[i] = dataFile[order[i]]
                newAnswOrder[i] = answerFile[order[i]]
        return newDataOrder,newAnswOrder

# This function calculates the probability that a word will occur given that the email provided is spam, or P(word | spam)
# In this case, A is the spam file and B is the datafile. The function returns a 1D vector with 57 elements, read as the
# P(wod | spam) for each possible word in the file.
def ProbBGivenA(A,B):
        numMeans=[0]*len(B[0])
        for i in range(len(B[0])):
                total=0
                counter=0
                for j in range(len(B)):
                        if A[j][0]==1:
                                total += 1
                                counter += B[j][i]
                numMeans[i]=counter/float(total)
        return numMeans

# Calculates the proportion of emails in the file that are spam.
def ProbSpam(A,B):
        counter=0
        for i in range(len(A)):
                if A[i][0]==1:
                        counter+=1
        return counter/float(len(A))

# Calculates the probability of every word over the whole file, so P(word). Returns a 1D matrix with 57 elements,
# where each element is the Pi of seeing word Xi
def ProbWord(A,B):
        numMeans=[0]*len(B[0])
        for i in range(len(B[0])):
                total=0
                counter=0
                for j in range(len(B)):
                        total += 1
                        counter += B[j][i]
                numMeans[i]=counter/float(total)
        return numMeans

# Given a matrix of evidence, this function calculates the P(spam | word) for each word. It returns a 1D 57 element matrix
# where each element Pi is the probability that an email is spam given that it has the word Xi
def BayesEachWord(PAB, A, B):
        bayesPerWord=[0]*len(B)
        numerator=[0]*len(B)
        for j in range(len(B)):
                bayesPerWord[j] = (PAB[j]*A)/B[j]
        return bayesPerWord

# Gets the total probability that an email is spam given all words that appear or do not appear in the email. Returns a single
# real value between 0 and 1 representing the probability that the email provided is spam.
def BayesAllWords(bayesPerWord, newEmail):
        num = 1
        den=1
        for i in range(len(bayesPerWord)):
                if newEmail[i] != 0:
                        num *= (bayesPerWord[i]*newEmail[i])
                        den *= ((1-bayesPerWord[i])*newEmail[i])
        return num/(num+den)
                        
# Splits the dataset and the corresponding class data into a series of N partitions provided with the numPartitions variable.
# The function returns an N element matrix with each element being a single fold of the dataset.
def SplitDataSet(dataset, answerSet, numPartitions):
        trainSize = int(len(dataset)/numPartitions)
        trainSet = [0]*numPartitions
        spamSet = [0]*numPartitions
        j=-1
        trainSet=[dataset[x:x+trainSize] for x in xrange(0, len(dataset)-1, trainSize)]
        spamSet=[answerSet[x:x+trainSize] for x in xrange(0, len(dataset)-1, trainSize)]
        return trainSet, spamSet

# Given a series of N-1 folds, this function trains the model on the provided N-1 training folds and tests the model against the remaining
# holdout fold. It prints the accuracy of the model against each fold, and returns a single dimensional list of the accuracies in order.
def TrainOnFolds(dataset,answerSet, holdout, holdoutAnswers):
        newDataSet = list(itertools.chain(*dataset))
        newAnswerSet = list(itertools.chain(*answerSet))
        HOA = list(itertools.chain(*holdoutAnswers))
        classifications=[0]*len(holdout)
        accuracy=0
        for i in range(len(holdout)):
                classifications[i]=int(round(BayesAllWords(BayesEachWord(ProbBGivenA(newAnswerSet,newDataSet),ProbSpam(newAnswerSet,newDataSet),ProbWord(newAnswerSet,newDataSet)),holdout[i])))

        for i in range(len(holdoutAnswers)):
                if classifications[i]==HOA[i]:
                        accuracy+=1
        print('Naive Bayes reported an accuracy of {0} for this fold'.format(accuracy/float(len(holdout))))
        return accuracy/float(len(holdout))

# Main method. Loads the appropriate data files, partititions the data (into 10 folds initially), and runs the model. The final print statement
# outputs the overall accuracy of the model.
def main():
        filename = 'spamdata_binary.txt'
        #Number of folds.
        splitRatio = 10
        initialDataset = LoadWordData(filename)
        filename='spamlabels.txt'
        initialAnswerSet = LoadWordData(filename)
        dataset,answerSet = ShuffleFile(initialDataset, initialAnswerSet)
        trainSet,newAnswerSet = SplitDataSet(dataset,answerSet,splitRatio)
        newObservation=dataset[2000]
        print(BayesAllWords(BayesEachWord(ProbBGivenA(answerSet,dataset),ProbSpam(answerSet,dataset),ProbWord(answerSet,dataset)),newObservation))
        x=ProbBGivenA(answerSet,dataset)
        for i in range(57):
                x[i]=log(x[i],10)
        accuracy=[0]*splitRatio
        for i in range(len(trainSet)):
                holdoutAnswers = newAnswerSet.pop(i)
                holdout = trainSet.pop(i)
                accuracy[i] = TrainOnFolds(trainSet,newAnswerSet,holdout,holdoutAnswers)
                newAnswerSet = [holdoutAnswers] + newAnswerSet
                trainSet = [holdout] + trainSet

        print('Overall accuracy of model: {0}'.format(sum(accuracy)/float(len(accuracy))))

main()
