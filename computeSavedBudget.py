from utils import *
print "DERP"
from runLotsFunction import * 
print "DERP2"
#from mlml import loadExamples
print "DERP3"
from random import sample
print "DERP4"
import numpy as np
print "DERP5"
from sklearn.cluster import SpectralClustering
print "DERP6"
from sklearn.semi_supervised import LabelSpreading
print "DERP7"

from pyrite import *
print "DERP8"


def runEverything():
    #dataset = 'data/breast-cancer-wisconsin-formatted.data'
    #dataset = 'data/data_banknote_authentication.txt'
    #dataset = 'data/seismicbumps.data'
    #dataset = 'data/eegeyestate.data'
    #dataset = 'data/sonar.data'
    #dataset = 'data/wdbc-formatted.data'
    #dataset = 'data/ad-formatted.data'
    #dataset = 'data/gisette.data'
    #dataset = 'data/hv.data'
    #dataset = 'data/hvnoise.data'
    #dataset = 'data/hv2.data'
    #dataset = 'data/spambase.data'
    #dataset = 'data/farm-ads-formatted.data'
    #dataset = open('data/winequality-5binary-red.data', 'r')
    #dataset = open('data/winequality-5binary-white.data', 'r') 
    #dataset = open('data/abalone_9binary.data', 'r')
    #dataset = open('data/processed.cleveland.data', 'r') 
    dataset = open('data/forestfires_0binary.data', 'r') 


    (instances, classes) = readData(dataset)
    numFeatures = len(instances[0])
    numInstances = len(instances)
    #budget = int(floor(len(instances) * 0.75))
    #budgetInterval = 50
 
    budget = int(floor(len(instances) * 0.5))
    #budget = 950
    budgetInterval = budget
    #budget = 50
    #budgetInterval = 50 

    
    print "Doing the clustering"
    #sc = SpectralClustering(n_clusters=2,
    #                        affinity = 'nearest_neighbors')

    #sc = SpectralClustering(n_clusters=2)
    

    #predictedLabels = sc.fit_predict(np.array(instances))

    #doPyriteSSL(instances, budget)
    doPyrite(instances, budget)
    
    """
    accuracy = 0.0
    inverseAccuracy = 0.0
    for (label, predictedLabel) in zip(classes, predictedLabels):
        if label == predictedLabel:
            accuracy += 1
        if not label == predictedLabel:
            inverseAccuracy += 1
    print "Correlations:"
    print accuracy / len(predictedLabels)
    print inverseAccuracy / len(predictedLabels)
    """
    """
    bestStrategy = getBestPolicy2(budget = budget, 
                                  budgetInterval = budgetInterval, 
                                  workerAccuracies = [1.0],
                                  numSimulations = 1000, 
                                  instances = instances,
                                  classes = classes,
                                  classifier = LRWrapper(),
                                  d = 0.5, numClasses = 2,
                                  writeToFile = False)
    """
def computeSavedBudget():

    #dataset = 'data/breast-cancer-wisconsin-formatted.data'
    #dataset = 'data/data_banknote_authentication.txt'
    #dataset = 'data/seismicbumps.data'
    dataset = 'data/eegeyestate.data'
    #dataset = 'data/sonar.data'
    #dataset = 'data/wdbc-formatted.data'
    #dataset = 'data/ad-formatted.data'
    #dataset = 'data/gisette.data'
    #dataset = 'data/hv.data'
    #dataset = 'data/hvnoise.data'
    #dataset = 'data/spambase.data'
    #dataset = 'data/farm-ads-formatted.data'

    (instances, classes) = readData(dataset)
    numFeatures = len(instances[0])
    numInstances = len(instances)
    budget = int(floor(len(instances) * 0.75))
    budgetInterval = 50


    averageAccuracies = pickle.load(open(
        "realdataresults/runlotsAccuracies%d,%d,%d,%d" % 
        (numFeatures,numInstances, budget, budgetInterval), 
        "rb"))


    #print averageAccuracies

    if dataset == 'data/hvnoise.data':
        nnAccuracies = pickle.load(open(
                "sc-nn-results/runlotsAccuracies%d,%d,%d,%dn" % 
                (numFeatures,numInstances, budget, budgetInterval), 
                "rb"))    
        rbfAccuracies = pickle.load(open(
                "sc-rbf-results/runlotsAccuracies%d,%d,%d,%dn" % 
                (numFeatures,numInstances, budget, budgetInterval), 
                "rb"))
    else:
        nnAccuracies = pickle.load(open(
                "sc-nn-results/runlotsAccuracies%d,%d,%d,%d" % 
                (numFeatures,numInstances, budget, budgetInterval), 
                "rb"))    
        rbfAccuracies = pickle.load(open(
                "sc-rbf-results/runlotsAccuracies%d,%d,%d,%d" % 
                (numFeatures,numInstances, budget, budgetInterval), 
                "rb"))


    budgets = range(budgetInterval, budget+1, budgetInterval)

    simNum = 1
    numSimulations = 10000
    extraBudgetRatios = []
    accLosses = []
    baselinePolicy = 3

    while simNum < numSimulations:
        print simNum
        #print budgets
        betterPolicyBudget = sample(budgets, 1)[0]
        print betterPolicyBudget
        #betterPolicyBudget = int(floor(len(instances) * 0.5)) + 1
        #betterPolicyBudget = 150

        if (max(nnAccuracies[(1.0, betterPolicyBudget)]) >
            max(rbfAccuracies[(1.0, betterPolicyBudget)])):
            betterPolicy = np.argmax(nnAccuracies[(1.0, betterPolicyBudget)])
        else:
            betterPolicy = np.argmax(rbfAccuracies[(1.0, betterPolicyBudget)])

        targetAccuracy = averageAccuracies[(1.0, 
                                            betterPolicyBudget)][betterPolicy]

        baselineAccuracy = averageAccuracies[(1.0, 
                                            betterPolicyBudget)][baselinePolicy]

        accLoss = baselineAccuracy - targetAccuracy
        accLosses.append(accLoss)
        print betterPolicy
        print targetAccuracy
        #print "HERE"
        requiredBudget = 1298921312
        for (j, b) in zip(range(len(budgets)), budgets):
            #print averageAccuracies[(1.0, b)][baselinePolicy]
            if averageAccuracies[(1.0, b)][baselinePolicy] >= targetAccuracy:
                requiredBudget = b
                break
        #print "HMM"
        print requiredBudget
        #print betterPolicyBudget

        if requiredBudget > budget:
            continue
        simNum += 1
        extraBudgetRatios.append(requiredBudget / betterPolicyBudget)

    #print extraBudgetRatios
    print np.average(extraBudgetRatios)
    print (1.96 * np.std(extraBudgetRatios)) / sqrt(numSimulations)
        

def computeAccuracyLoss():
    #dataset = 'data/breast-cancer-wisconsin-formatted.data'
    #dataset = 'data/data_banknote_authentication.txt'
    #dataset = 'data/seismicbumps.data'
    #dataset = 'data/eegeyestate.data'
    #dataset = 'data/sonar.data'
    #dataset = 'data/wdbc-formatted.data'
    #dataset = 'data/ad-formatted.data'
    #dataset = 'data/gisette.data'
    #dataset = 'data/hv.data'
    #dataset = 'data/hvnoise.data'
    dataset = 'data/spambase.data'
    #dataset = 'data/farm-ads-formatted.data'

    (instances, classes) = readData(dataset)
    numFeatures = len(instances[0])
    numInstances = len(instances)
    budget = int(floor(len(instances) * 0.75))
    budgetInterval = 50

    averageAccuracies = pickle.load(open(
        "realdataresults/runlotsAccuracies%d,%d,%d,%d" % 
        (numFeatures,numInstances, budget, budgetInterval), 
        "rb"))

    if dataset == 'data/hvnoise.data':
        nnAccuracies = pickle.load(open(
                "sc-nn-results/runlotsAccuracies%d,%d,%d,%dn" % 
                (numFeatures,numInstances, budget, budgetInterval), 
                "rb"))    
        rbfAccuracies = pickle.load(open(
                "sc-rbf-results/runlotsAccuracies%d,%d,%d,%dn" % 
                (numFeatures,numInstances, budget, budgetInterval), 
                "rb"))
    else:
        nnAccuracies = pickle.load(open(
                "sc-nn-results/runlotsAccuracies%d,%d,%d,%d" % 
                (numFeatures,numInstances, budget, budgetInterval), 
                "rb"))    
        rbfAccuracies = pickle.load(open(
                "sc-rbf-results/runlotsAccuracies%d,%d,%d,%d" % 
                (numFeatures,numInstances, budget, budgetInterval), 
                "rb"))


    budgets = range(budgetInterval, budget+1, budgetInterval)

    simNum = 0
    numSimulations = 1
    extraBudgetRatios = []
    betterPolicyAccuracyLosses = []
    baselineAccuracyLosses= []
    accLosses = []
    baselinePolicy = 2

    while simNum < numSimulations:
        #print simNum
        #print budgets
        betterPolicyBudget = sample(budgets, 1)[0]
        #print betterPolicyBudget
        #betterPolicyBudget = int(floor(len(instances) * 0.5)) + 1
        betterPolicyBudget = 1150

        if (max(nnAccuracies[(1.0, betterPolicyBudget)]) >
            max(rbfAccuracies[(1.0, betterPolicyBudget)])):
            betterPolicy = np.argmax(nnAccuracies[(1.0, betterPolicyBudget)])
        else:
            betterPolicy = np.argmax(rbfAccuracies[(1.0, betterPolicyBudget)])

        print betterPolicy
        targetAccuracy = averageAccuracies[(1.0, 
                                            betterPolicyBudget)][betterPolicy]

        baselineAccuracy = averageAccuracies[(1.0, 
                                            betterPolicyBudget)][baselinePolicy]

        baselineAccuracyLosses.append(max(
                averageAccuracies[(1.0, betterPolicyBudget)]) -baselineAccuracy)

        betterPolicyAccuracyLosses.append(max(
                averageAccuracies[(1.0, betterPolicyBudget)]) -targetAccuracy)
        accLoss = baselineAccuracy - targetAccuracy
        accLosses.append(accLoss)
        #print betterPolicy
        #print targetAccuracy

        simNum += 1

    #print extraBudgetRatios
    print np.average(accLosses)
    print (1.96 * np.std(accLosses)) / sqrt(len(accLosses))
    print np.average(baselineAccuracyLosses)    
    print np.average(betterPolicyAccuracyLosses)


print "RUNNING EVERYTHING"    
runEverything()
#computeAccuracyLoss()
#computeSavedBudget()
