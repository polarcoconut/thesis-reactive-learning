from random import random, sample
from scipy.misc import comb
from scipy.sparse import dok_matrix
from math import log
import numpy as np
import math
#from scipy.optimize import curve_fit, leastsq, fsolve, newton, brentq
from scipy.optimize import leastsq, fsolve, newton, brentq
import numpy as np
#import matplotlib.pyplot as plt

def computeStats(allVotes):
    numExamplesRelabeled = 0
    numTimesRelabeled = 0
    sampledIndices = []

    #Get some statistics about the relabeling
    for task in allVotes.keys():
        if task == -1:
            continue
        votes = allVotes[task]
        totalVotes = 0.0
        for vote in votes:
            totalVotes += vote
        if totalVotes > 1:
            numTimesRelabeled += totalVotes
            numExamplesRelabeled += 1


    return ([], numExamplesRelabeled, numTimesRelabeled)

def calcExpectedError(tasks, classifier):
    predictions = classifier.predict_proba(tasks)
    expectedError = 0.0
    for prediction in predictions:
        expectedError += 2.0 * prediction[0] * prediction[1]
    
    return expectedError
    
#Computes the probability of the label given the votes
def calcBayesProbability(votes, accuracy, prior=[0.5,0.5]):
    """
    pZero = 1.0
    pOne = 1.0

    for zeroVotes in range(votes[0]):
        pZero *= accuracy
        pOne *= (1.0 -accuracy)
    for oneVotes in range(votes[1]):
        pOne *= accuracy
        pZero *= (1.0 - accuracy)
    """

    if prior[0] < 0.00001 or prior[1] < 0.00001:
        return prior

    pZero = 0.0
    pOne = 0.0

    for zeroVotes in range(votes[0]):
        pZero +=  log(accuracy)
        pOne += log(1.0 -accuracy)
    for oneVotes in range(votes[1]):
        pOne += log(accuracy)
        pZero += log(1.0 - accuracy)

    pZero += log(prior[0])
    pOne += log(prior[1])

    normalizationConstant = np.logaddexp(pZero,pOne)

    pZero -= normalizationConstant
    pOne -= normalizationConstant

    pZero = np.exp(pZero)
    pOne = np.exp(pOne)

    if abs((pOne + pZero)- 1.0) > 0.00001:
        print "WRONG WRONG WRONG"
        print pOne + pZero
        print pOne
        print pZero

    if pOne+pZero == 0:
        print "Caculating Bayes"
        print pOne + pZero
        print pOne
        print pZero
        print votes
    return (pZero, pOne)
    #return (pZero / (pOne + pZero), pOne / (pOne + pZero))
    
def update(task, votes, classifier, accuracy):
    #print "HUH"
    (pZero, pOne) = calcBayesProbability(votes, accuracy)
    aggregatedLabel = np.argmax(votes)
    #print aggregatedLabel
    #print (pZero, pOne)
    maxVotes = max(votes)
    pMax = max(pZero, pOne)
    #print pMax
    #print int(pMax * 100.0) - 1
    #print "BLOB"
    classifier.update([task], [aggregatedLabel], [pMax])
    

def retrain(labels, classifier, 
            bayesOptimal = False, accuracy = None, goldLabels = None):
    trainingTasks = []
    trainingLabels = []
    trainingWeights = []
    trainingDict = {}
    subsetGoldLabels = []

    #print "retraining"
    #print len(labels.keys())
    #for (task, (trues, falses)) in zip(allTasks, labels):
    for task in labels.keys():
        if task == -1:
            continue
    #for (i, task, votes) in zip(range(len(allTasks)), allTasks, labels):
        #if trues == 0 and falses == 0:
        #    continue
        #if trues == falses:
        #    continue
        votes = labels[task]
        allZero = True
        for vote in votes:
            if vote != 0:
                allZero = False
                break
        if allZero:
            continue
        if votes[0] == votes[1]: 
            continue
        #if votes[0] + votes[1] > 50:
        #    print "Retraining"
        #    print votes
        #    print np.argmax(votes)
        #print "Another task"
        #print votes
        #if bayesOptimal:
        if False:
            (pZero, pOne) = calcBayesProbability(votes, accuracy)
            #Sometimes there will be a third vote, for the number of times
            # we want to add this point.
            aggregatedLabel = np.argmax(votes[0:2])
            #print (pZero, pOne)
            maxVotes = max(votes[0:2])
            pMax = max(pZero, pOne)
            #print pMax
            #print int(pMax * 100.0) - 1
            if False: #classifier has the ability to take instance weights
                trainingLabels.append(aggregatedLabel)
                trainingTasks.append(task)
                trainingWeights.append(pMax)
            else:
                #for i in range(int(pMax * 100.0) - 1):
                for i in range(maxVotes):
                    #print maxVotes
                    #print i
                    trainingLabels.append(aggregatedLabel)
                    trainingTasks.append(task)
            #for i in range(int(pOne * 100.0)):
            #for i in range(votes[1]):
            #    trainingLabels.append(1)
            #    trainingTasks.append(task)
            #print trainingLabels
        else:
            #if np.argmax(votes) == 0:
            #    trainingLabels.append(-1)
            #else:
            #    trainingLabels.append(1)
            if goldLabels != None:
                subsetGoldLabels.append(goldLabels[i])

            if len(votes) > 2:
                for k in range(votes[2]):
                    trainingLabels.append(np.argmax(votes[0:2]))
                    trainingTasks.append(task)
                    trainingWeights.append(1.0)
            else:
                trainingLabels.append(np.argmax(votes))
                trainingTasks.append(task)
                trainingWeights.append(1.0)
            #trainingDict[task] = np.argmax(votes)
        
        """
        if np.argmax(votes) == 1:
            plt.plot(task[0], task[1], 'rx')
        elif np.argmax(votes) == 0:
            plt.plot(task[0], task[1], 'bo')
        """


    #SANITY CHECK IF YOU HAVE THE GOLD LABELS
    if goldLabels != None:
        numCorrect = 0.0
        for (trainingLabel, goldLabel) in zip(trainingLabels, subsetGoldLabels):
            if trainingLabel == goldLabel:
                numCorrect += 1

        print numCorrect / len(trainingLabels)
    #print trainingLabels
    #print "NUMBER OF TRAINING TASKS"
    #print len(trainingTasks)
    #print len(trainingLabels)
    #print trainingLabels
    #print trainingTasks
    #plt.show()
    
    #print "Number of training tasks"
    #print len(trainingTasks)
    
    #DONT DO THIS IN REAL LIFE
    #trainingTasks = []
    #trainingLabels = []
    #trainingWeights = []
    #items = sorted(trainingDict.items())
    #for (task, label) in items:
    #    trainingTasks.append(task)
    #    trainingLabels.append(label)
    #    trainingWeights.append(1.0)

    classifier.retrain(trainingTasks, trainingLabels, trainingWeights)
    #classifier.retrain(trainingTasks, trainingLabels)
    
    """
    weights = classifier.classifier.coef_
    print weights
    intercept = classifier.classifier.intercept_
    print intercept

    xs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ys = []
    for x in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        print (x, (intercept[0] + weights[0][0]*x)/ (-1 * weights[0][1]) )
        ys.append((intercept[0] + weights[0][0]*x)/ (-1 * weights[0][1]) )
    plt.plot(xs, ys, 'g--')
    plt.show()
    """
    
    return (trainingTasks, trainingLabels)

def retrainSSL(allTasks, labels, ssl):
    trainingLabels = []
    #subsetGoldLabels = []

    unlabeledTasks = []
    labeledTasks = []
    labeledTaskLabels = []

    for (i, task, votes) in zip(range(len(labels)), allTasks[0:len(labels)],
                                labels):
        allZero = True
        for vote in votes:
            if vote != 0:
                allZero = False
                break
        if allZero:
            unlabeledTasks.append(task)
            trainingLabels.append(-1)
        else:
            labeledTasks.append(task)
            aggregatedLabel = np.argmax(votes)
            labeledTaskLabels.append(aggregatedLabel)
            trainingLabels.append(aggregatedLabel)

    print "WHAT ABOUT HERE"
    #This is here because allTasks includes the validation and testingTasks for some reason
    
    for i in range(len(labels), len(allTasks)):
        trainingLabels.append(-1)
    

    ssl.fit(allTasks, trainingLabels)
    print "FITTED"
    
    #print ((unlabeledTasks, ssl.predict(unlabeledTasks)),
    #        (labeledTasks, labeledTaskLabels))

    return (ssl.predict(allTasks),
            (labeledTasks, labeledTaskLabels))
            
    #return ssl.predict(allTasks)

def labelAccuracy(trainingTasks, state, classes):
    trainingTasks = []
    trainingLabels = []

    numberCorrect = 0.0
    numberTotal = 0.0

    for ((trues, falses), c) in zip(state, classes):
        if trues == 0 and falses == 0:
            continue
        if trues == falses:
            continue
        if trues >= falses:
            label = 1
        else:
            label = 0

        
        #if trues > falses:
        #    label = 1
        #elif falses > trues:
        #    label = 0
        #else:
        #    if random() < 0.5:
        #        label = 1
        #    else:
        #        label = 0
        
        if label == c:
            numberCorrect += 1.0
        numberTotal += 1.0

    return numberCorrect / numberTotal


#get the most unsure example based on peng's model
def getRelabelingPolicy(taskIndices, state, acc):

    bestExamples = []
    bestV = 100


    #Generalized round-robin
    for i in taskIndices:
        (trues, falses) = state[i]
        if (trues + falses) < bestV:
            bestV = trues + falses
            bestExamples = [i]
        elif (trues + falses) == bestV:
            bestExamples.append(i)
        else:
            continue
            
    """
    #label uncertainty
    for i in taskIndices:
        (trues, falses) = state[i]
        pTrue = 1.0
        pFalse = 1.0
        if trues > 0:
            pTrue *= (acc ** trues)
            pFalse *= (1.0 -acc) ** trues
        if falses > 0: 
            pTrue *= (1.0-acc) ** falses
            pFalse *= (acc ** falses)

        normalization = pTrue + pFalse
        pTrue /= normalization
        pFalse /= normalization
        

        if min(pTrue, pFalse) < bestV:
            bestV = min(pTrue, pFalse)
            bestExamples = [i]
        elif min(pTrue, pFalse) == bestV:
            bestExamples.append(i)
      
    """
    
    return sample(bestExamples,1) [0]



def computeQualityCont(numLabels, x0, y0, c, k):
    return (c / (1 + np.exp(-k * (numLabels - x0))))

#numLabels is numLabels per task.
def computeQuality(acc, numLabels):
    numLabels = int(numLabels)
    #print "num labels"
    #print numLabels
    #if numLabels % 2 == 0:
    #    numLabels += 1

    k = (numLabels - 1) / 2

    s = 0.0

    denom = 0.0

    #weights = [1.0, 0.3, 0.3, 1.0]
    """
    for i in range(k+1):
        weights.append(((1.0 * i) / numLabels))
    for i in range(k+1, numLabels+1):
        j = numLabels - i
        weights.append(((1.0 * j) / numLabels))
    """
    #print weights

    for i in range(k+1):
        s += comb(numLabels, i) * (acc ** (numLabels - i)) * ((1.0 - acc) ** i)

    if numLabels % 2 == 0:
        i = k+1
        s += 0.5 * comb(numLabels, i) * (acc ** (numLabels - i)) * ((1.0 - acc) ** i)
    
    #for i in range(numLabels+1):
    #    denom += weights[i] * comb(numLabels, i) * (acc ** (numLabels - i)) * ((1.0 - acc) ** i)
        
    #if numLabels % 2 == 0:
    #    s += 0.5 * comb(numLabels, k+1) * (acc ** (numLabels - (k+1))) * ((1.0 - acc) ** (k+1))
    return s 


def readBagOfWords(f, numFeatures, numExamples):
    classes = []
    instances = dok_matrix((numExamples, numFeatures))
    for (row, line) in zip(range(numExamples), f):
        line = line.split(' ')
        classes.append(int(line[0]))
        for token in line[1:]:
            token = token.split(':')
            feature = int(token[0])
            value = int(token[1])
            instances[row, feature] = value
    
    return instances

        

def calcPAC2(e, m, tau, d, vc, a, noise, budget):
    N = max((4.0 / e * log(2.0 / d, 2.0), 8.0 * vc / e * log(8.0 * vc / e, 2)))
    
    term1 = 2.0 / (((e/2.0) ** 2) * ((1.0-2.0 * noise) ** 2))
    term2 = log((2.0 * N) / (2.0 * d / 3.0))
    
    return term1 * term2 - m

def calcPAC(e, k, m, tau, d, vc, a, noise, budget):
   
    #print "M"
    #print m
 

    #print "E"
    #print e

    if e < 0:
        e = 0.000001
    if e > 1.0:
        e = 0.999999

    #If we get m examples, then there is budget - m  left to split among the
    # examples.
    if m < 1:
        m = 1
    m = int(m)
    #print m
   # print (m, tau, d, e, vc, a, noise, budget)

    #print "Terms"
    firstTerm = 1.0 / ((tau ** 2) * (e ** 2) * ((1.0 - (2.0 * noise)) ** 2))
    #print firstTerm
    secondTerm = log(1.0 / e) ** 2
    #print secondTerm
    thirdTerm = vc + (vc * log(1.0 / e) * log(log(1.0 / e)))
    #print thirdTerm
    fourthTerm = log((1.0 / (tau * e * (1.0 - (2.0 * noise)))) * log(1.0/e))
    #print fourthTerm
    term = (firstTerm * (log(1.0/e) ** 2) * (thirdTerm * fourthTerm + log(1.0/d))) / k - m
    #print "TERM"
    #print term

    return term


def computeLogisticCurve(acc, maxX):
    xs = range(1, maxX, 2)
    xs2 = range(-1 * maxX + 2, 1, 2)
    ys = []
    ys2 = []
    for numLabels in xs:
        q = computeQuality(acc, numLabels)
        ys.append(q)


    print xs
    print ys
    for q in ys:
        ys2.insert(0, 1.0-q)
 


    popt, pcov = curve_fit(computeQualityCont, xs2 + xs, ys2 + ys) 

    (x0, y0, c, k) = popt
    
    return (x0, y0, c, k)
    
"""
def simLabel(d, g, correctLabel):
    a = (1.0 / 2.0) * (1.0 + ((1.0 - d) ** g))
    r = random()
    if r < a:
        return correctLabel
    else:
        return 1 - correctLabel
"""

def simLabel(d, g, correctLabel, incorrectLabels):
    a = (1.0 / 2.0) * (1.0 + ((1.0 - d) ** g))
    #print "accuracy"
    #print a

    r = random()
    if r < a:
        return correctLabel
    else:
        return sample(incorrectLabels,1)[0]



def findLocalMin(xs, ys):


    dipX = 98019231
    dipY = 120893
    lastX = -10293801
    lastY = -109820391
    goesDown = False

    for (x, y) in zip(xs, ys):
        if y < lastY:
            goesDown = True
            dipX = x
            dipY = y
        lastX = x
        lastY = y

    if goesDown:
        return dipX
    else:
        return xs[0]


def readData(filename):
    readdatafile = filename
    #readdatafile = open(filename, 'r')

    instances = []
    classes = []
    numberTrue = 0
    for line in readdatafile:
        line = line.split(',')
        instances.append([float(line[i]) for i in range(len(line) - 1)])
        classes.append(int(line[len(line) - 1]))
        
    readdatafile.close()

    return (instances, classes)
