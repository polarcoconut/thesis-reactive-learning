from random import sample, randint, random
import cPickle as pickle

from utils import *
from runLotsFunction import * 

from math import ceil
#import matplotlib.pyplot as plt 
import numpy as np
from itertools import product

from sklearn import tree
from sklearn.cluster import SpectralClustering

from data.makedata import makeGaussianData, makeTriGaussianData
from scipy.stats import spearmanr
from pyrite import *
#from sslrun import *

#classifiers = [LRWrapper(), 
#               DTWrapper(),
#               SVMWrapper(),
#               RFWrapper(),
#               NNWrapper(),
#               PerceptronWrapper()]


classifiers = [DTWrapper()]

classifierIndices = [0,1,2,3,4,5,6]
classifierIndices = [0]
classifiers = zip(classifiers, classifierIndices)

#workerAccuracies = [3.3, 1.7, 1.0, 0.5, 0.15]
#workerAccuracies = [3.3, 1.7, 1.0, 0.5, 0.0]
workerAccuracies = [1.0]
#workerAccuracyPercentages = [.55, .65, .75, .85, .95]
#workerAccuracyPercentages = [.55, .65, .75, .85, 1.0]
workerAccuracyPercentages = [0.75]
workerAccuracyHash = {}
index = 0
for workerAccuracyPercentage in workerAccuracyPercentages:
    workerAccuracyHash[workerAccuracyPercentage] = index 
    index += 1

workerAccuracies = zip(workerAccuracies, workerAccuracyPercentages)

#budgets = range(50, 500, 50)
#budgets = [100]
budgets = range(50, 2000, 50)


def loadExamples(numExamples, maxBudget, budgetInterval, 
                 numFeatures, strategies):
    examples = []
    labels = []
    budgets = range(budgetInterval, maxBudget+budgetInterval, 
                    budgetInterval)
    trainingPolicy = np.zeros((len(workerAccuracies), len(budgets))) + -1
    #print sample([(x,y) for x in workerAccuracies for y in budgets],
    #             numExamples)
    for ((workerAccuracy, workerAccuracyPercentage),
         budget) in sample([(x,y) for x in workerAccuracies for y in budgets],
                           numExamples):

        #for i in range(numExamples):    
        #(workerAccuracy, 
        # workerAccuracyPercentage) = sample(workerAccuracies, 1)[0]

        #budget = sample(budgets, 1)[0]
        #print (workerAccuracy, budget)
        #print len(budgets) - budget / budgetInterval
        #print strategies
        try:
            bestStrategy = strategies[2, budget/budgetInterval-1] 
        except:
            bestStrategy = strategies[0, budget/budgetInterval-1]
        #bestStrategy = strategies[workerAccuracyHash[workerAccuracyPercentage], 
#                                   budget / budgetInterval - 1]
        #print bestStrategy
    
        example = [0] * 10
        example[0] = budget
        example[1] = workerAccuracyPercentage
        example[2] = numFeatures
        classifierIndex = 0
        example[classifierIndex + 3] = 1
    
        examples.append(example)
        label = bestStrategy
        """
        if label > 0:
            label = 1
        else:
            label = 0
        """
        label += 1
        labels.append(label)

        #trainingPolicy[workerAccuracyHash[workerAccuracyPercentage], 
        #       budget/500 - 1] = label
        #print MLMLClassifier.predict(example)

    """
    fig, ax = plt.subplots()
    ax.matshow(trainingPolicy, cmap = plt.cm.gray_r)
    ax.set_xticklabels(budgets)
    plt.xticks(range(len(budgets)), budgets)
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("Budget")
    ax.set_yticklabels([55, 65, 75, 85, 95])
    plt.yticks(range(len(workerAccuracies)), [55,65,75,85,95])
    ax.set_ylabel("Worker Accuracy")
    plt.show()
    """
    return (examples, labels)

def genExamples(numExamples, numSimulations):
    examples = []
    labels = []
    for i in range(numExamples):
        print "GENERATING EXAMPLE %d" % i
        numFeatures = randint(1, 10)
        #numFeatures = 2
        (classifier, classifierIndex) = sample(classifiers, 1)[0]
        (workerAccuracy, 
         workerAccuracyPercentage) = sample(workerAccuracies, 1)[0]
        budget = sample(budgets, 1)[0]

        print budget
        print numFeatures
        
        bestStrategy = getBestPolicy(budget = budget, budgetInterval = budget, 
                                     workerAccuracies = [workerAccuracy],
                                     numSimulations = numSimulations, 
                                     numFeatures = numFeatures, 
                                     numExamples = int(ceil(budget / 0.8)),
                                     classifier = classifier,
                                     d = 0.5, numClasses = 2)

        
        example = [0] * 10
        example[0] = budget
        example[1] = workerAccuracyPercentage
        example[2] = numFeatures
        example[classifierIndex + 3] = 1
        

        examples.append(example)
        label = bestStrategy[0,0]
        if label > 0:
            label = 1
        else:
            label = 0
        labels.append(label)

    return (examples, labels)


def mlml3():
    
    #generate a random gaussian dataset
    numDatasets = 39
    policies = [0,1,2,3]
    accuracyLosses = [[] for i in range(len(policies))]
    pyriteAccuracyLosses = []
    adaptivePyriteAccuracyLosses = []

    for numDataset in range(numDatasets):
        print "DATASET NUMBER"
        print numDataset
        numFeatures = sample([10, 50, 90], 1)[0] 
        budget = sample(range(50, 350, 50),1)[0]

        #numFeatures = sample([16], 1)[0] 
        #budget = 100

        numExamples = int(ceil(budget / 0.8))
        (instances, classes) = makeTriGaussianData(numExamples, numFeatures, 0, 
                                                noise=False, numClasses=2, 
                                                skew=1.0, 
                                                f = None, randomData=True,
                                                writeToFile = False)
        
        print "DOING PYRITE SSL"
        predictedBestStrategy = doPyriteSSL(instances, budget)
        print "DOING ADAPTIVE PYRITE"
        (adaptiveStrategy, adaptiveMeans) = getBestPolicySSL(budget = budget, 
                                    budgetInterval = budget, 
                                    workerAccuracies = [1.0],
                                    numSimulations = 100, 
                                    instances = instances,
                                    classes = classes,
                                    classifier = LRWrapper(),
                                    d = 0.5, numClasses = 2,
                                    policies = [predictedBestStrategy],
                                    writeToFile = False)

        #Get the actual best policy
        (actualBestStrategy,
         actualAccuracies) = getBestPolicy2(budget = budget, 
                                             budgetInterval = budget, 
                                             workerAccuracies = [1.0],
                                             numSimulations = 100, 
                                             instances = instances,
                                             classes = classes,
                                             classifier = LRWrapper(),
                                             d = 0.5, numClasses = 2,
                                             policies = policies,
                                             writeToFile = False)
        actualBestStrategy = actualBestStrategy[0,0]
         

        pyriteAccuracyLoss = (
            actualAccuracies[(1.0, budget)][int(actualBestStrategy)] -
            actualAccuracies[(1.0, budget)][int(predictedBestStrategy)])
        
        pyriteAccuracyLosses.append(pyriteAccuracyLoss)
        
        adaptivePyriteAccuracyLoss = (
            actualAccuracies[(1.0, budget)][int(actualBestStrategy)] -
            adaptiveMeans[(1.0, budget)][int(predictedBestStrategy)])
        
        adaptivePyriteAccuracyLosses.append(adaptivePyriteAccuracyLoss)
        
        for policy in policies:
            accuracyLosses[policy].append((
                    actualAccuracies[(1.0, budget)][int(actualBestStrategy)] -
                    actualAccuracies[(1.0, budget)][policy]))

    print np.average(pyriteAccuracyLosses)
    print np.sum(pyriteAccuracyLosses)
    print np.std(pyriteAccuracyLosses)


    print np.average(adaptivePyriteAccuracyLosses)
    print np.sum(adaptivePyriteAccuracyLosses)
    print np.std(adaptivePyriteAccuracyLosses)



    print [np.average(x) for x in accuracyLosses]
    print [np.sum(x) for x in accuracyLosses]
    print [np.std(x) for x in accuracyLosses]




def mlml2():

    print "Begin Unsupervised Learning"
    #sc = SpectralClustering(n_clusters=2,
    #                        affinity = 'nearest_neighbors')

    sc = SpectralClustering(n_clusters=2)

    #sc = SpectralClustering(n_clusters=2,
    #                        eigen_solver = 'amg')

    
    files = ['data/breast-cancer-wisconsin-formatted.data',
             'data/data_banknote_authentication.txt',
             'data/seismicbumps.data',
             'data/eegeyestate.data',
             'data/sonar.data',
             'data/wdbc-formatted.data',
             'data/ad-formatted.data',
             'data/gisette.data',
             'data/hv.data',
             'data/hvnoise.data',
             'data/spambase.data',
             'data/farm-ads-formatted.data']


    #(instances, classes) = readData(
    #    'runLotsData/50,2500.data')

    #(instances, classes) = readData(
    #    'runLotsData/100,625.data')


    #(instances, classes) = readData(
    #    'data/breast-cancer-wisconsin-formatted.data')
    #(instances, classes) = readData(
    #    'data/data_banknote_authentication.txt')
    #(instances, classes) = readData(
    #    'data/seismicbumps.data')
    #(instances, classes) = readData(
    #    'data/eegeyestate.data')
    (instances, classes) = readData(
        'data/sonar.data')
    #(instances, classes) = readData(
    #    'data/wdbc-formatted.data')
    #(instances, classes) = readData(
    #    'data/ad-formatted.data')
    #(instances, classes) = readData(
    #    'data/gisette.data')
    #(instances, classes) = readData(
    #    'data/hv.data')
    #(instances, classes) = readData(
    #    'data/hvnoise.data')
    #(instances, classes) = readData(
    #    'data/spambase.data')
    #(instances, classes) = readData(
    #    'data/farm-ads-formatted.data')




    #print instances[0:10]
    predictedLabels = sc.fit_predict(np.array(instances))

    """
    predictedLabels = []
    for instance in instances:
        if random() < 0.65:
            predictedLabels.append(1)
        else:
            predictedLabels.append(0)
    
    """
    #Check the correlation between labels and predicted labels
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
    print spearmanr(classes, predictedLabels)
    """
    #raise Exception 

    #budget = int(floor(len(instances)/ 2))
    budget = 50
    bestStrategy = getBestPolicy2(budget = budget, 
                                 budgetInterval = budget, 
                                 workerAccuracies = [1.0],
                                 numSimulations = 1000, 
                                 instances = instances,
                                 classes = predictedLabels,
                                 classifier = LRWrapper(),
                                 d = 0.5, numClasses = 2,
                                  writeToFile = False)
    
    print bestStrategy
    #Check the correlation between labels and predicted labels
    accuracy = 0.0
    inverseAccuracy = 0.0
    numTruesBronze = 0.0
    numTruesGold = 0.0
    for (label, predictedLabel) in zip(classes, predictedLabels):
        if label == predictedLabel:
            accuracy += 1
        if not label == predictedLabel:
            inverseAccuracy += 1
        if predictedLabel == 1:
            numTruesBronze += 1
        if label == 1:
            numTruesGold += 1
    print "Correlations:"
    print accuracy / len(predictedLabels)
    print inverseAccuracy / len(predictedLabels)
    print spearmanr(classes, predictedLabels)
    print "Bronze Skew:"
    print numTruesBronze / len(predictedLabels)
    print "Gold Skew:"
    print numTruesGold / len(predictedLabels)

def mlml(numTrainingExamples, numTestingExamples):

    MLMLClassifier = SVMWrapper()
    #MLMLClassifier = RFWrapper()


    print "GENERATING TRAINING EXAMPLES"
    """
    (trainingExamples, trainingLabels) = genExamples(numTrainingExamples,
                                                     100)
    pickle.dump((trainingExamples, trainingLabels),
                open('mlmldata/trainingData%d,4' % numTrainingExamples,
                     "wb"))
    """
    numTotalExamples = numTrainingExamples + numTestingExamples

    
    trainingExamples = []
    trainingLabels = []

    (trainingExamplesT, trainingLabelsT) = loadExamples(
        1, 350, 350, 9,
        pickle.load(open('results/runlotsStrategies9,438,350,350')))
    trainingExamples = trainingExamples + trainingExamplesT
    trainingLabels = trainingLabels + trainingLabelsT

    (trainingExamplesT, trainingLabelsT) = loadExamples(
        1, 686, 686, 4, 
        pickle.load(open('results/runlotsStrategies4,858,686,686')))
    trainingExamples = trainingExamples + trainingExamplesT
    trainingLabels = trainingLabels + trainingLabelsT
    (trainingExamplesT, trainingLabelsT) = loadExamples(
        1, 1292, 1292, 18,
        pickle.load(open('results/runlotsStrategies18,1615,1292,1292')))
    trainingExamples = trainingExamples + trainingExamplesT
    trainingLabels = trainingLabels + trainingLabelsT
    (trainingExamplesT, trainingLabelsT) = loadExamples(
        1, 7490, 7490, 14,
        pickle.load(open('results/runlotsStrategies14,9363,7490,7490')))
    trainingExamples = trainingExamples + trainingExamplesT
    trainingLabels = trainingLabels + trainingLabelsT
    (trainingExamplesT, trainingLabelsT) = loadExamples(
        1, 104, 104, 60,
        pickle.load(open('results/runlotsStrategies60,130,104,104')))
    trainingExamples = trainingExamples + trainingExamplesT
    trainingLabels = trainingLabels + trainingLabelsT
    (trainingExamplesT, trainingLabelsT) = loadExamples(
        1, 285, 285, 30,
        pickle.load(open('results/runlotsStrategies30,357,285,285')))
    trainingExamples = trainingExamples + trainingExamplesT
    trainingLabels = trainingLabels + trainingLabelsT
    (trainingExamplesT, trainingLabelsT) = loadExamples(
        1, 303, 303, 100,
        pickle.load(open('results/runlotsStrategies100,379,303,303')))
    trainingExamples = trainingExamples + trainingExamplesT
    trainingLabels = trainingLabels + trainingLabelsT
    (trainingExamplesT, trainingLabelsT) = loadExamples(
        40, 2000, 50, 50,
        pickle.load(open('results/runlotsStrategies50,2500,2000,50')))
    trainingExamples = trainingExamples + trainingExamplesT
    trainingLabels = trainingLabels + trainingLabelsT
    print trainingLabels
    (trainingExamplesT, trainingLabelsT) = loadExamples(
        40, 2000, 50, 16,
        pickle.load(open('results/runlotsStrategies16')))
    trainingExamples = trainingExamples + trainingExamplesT
    trainingLabels = trainingLabels + trainingLabelsT
    (trainingExamplesT, trainingLabelsT) = loadExamples(
        40, 2000, 50, 100,
        pickle.load(open('results/runlotsStrategies100')))
    trainingExamples = trainingExamples + trainingExamplesT
    trainingLabels = trainingLabels + trainingLabelsT



    testingExamples = trainingExamples[numTrainingExamples:numTotalExamples]
    testingLabels = trainingLabels[numTrainingExamples:numTotalExamples]
#    MLMLClassifier.retrain(trainingExamples[0:numTrainingExamples], 
#                           trainingLabels[0:numTrainingExamples])

    MLMLClassifier.retrain(trainingExamples, trainingLabels)


    print "TRAINING!"
    print trainingLabels

    print "TESTING"
    print MLMLClassifier.predict([1180, 0.75, 1558, 1, 0, 0, 0, 0, 0, 0])
    print MLMLClassifier.predict([3000, 0.75, 5000, 1, 0, 0, 0, 0, 0, 0])
    print MLMLClassifier.predict([2072, 0.75, 54877, 1, 0, 0, 0, 0, 0, 0]) 
    print MLMLClassifier.predict([50, 0.75, 200, 1, 0, 0, 0, 0, 0, 0])
    print "GENERATING TESTING EXAMPLES"
    """
    (testingExamples, testingLabels) = genExamples(numTestingExamples,
                                                   100)
    pickle.dump((testingExamples, testingLabels),
                open('mlmldata/testingData%d,4' % numTestingExamples,
                     "wb"))
    """

    #(testingExamples, testingLabels) = loadExamples(
    #    100, 20000, 500, 
    #    pickle.load(open('results/runlotsStrategiesDT4,25000')))
    

    #SCORE
    print "MLML accuracy"

    
    policy = np.zeros((len(workerAccuracies), len(budgets)))
    for (workerAccuracyPercentage, budget) in (
        (x, y) for x in workerAccuracyPercentages for y in budgets):
        example = [0] * 10
        example[0] = budget
        example[1] = workerAccuracyPercentage
        example[2] = 50
        example[3] = 1
        #print example
        policy[workerAccuracyHash[workerAccuracyPercentage], 
               budget/50 - 1] = MLMLClassifier.predict(
            example)
        #print MLMLClassifier.predict(example)
    print policy

    """
    fig, ax = plt.subplots()
    ax.matshow(policy, cmap = plt.cm.gray_r)
    ax.set_xticklabels(budgets)
    plt.xticks(range(len(budgets)), budgets)
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("Budget")
    ax.set_yticklabels([55, 65, 75, 85, 95])
    plt.yticks(range(len(workerAccuracies)), [55,65,75,85,95])
    ax.set_ylabel("Worker Accuracy")
    plt.show()
    """
    print MLMLClassifier.score(testingExamples, testingLabels)
    #print MLMLClassifier.classifier.feature_importances_
    #print MLMLClassifier.classifier.estimators_

    #tree.export_graphviz(MLMLClassifier.classifier.estimators_[0],
    #                     out_file='tree.dot')
    #BASELINES:
    #Only unilabel
    print "Unilabeling Accuracy"
    unilabelAccuracy = 0.0
    for testingLabel in testingLabels:
        if testingLabel == 0:
            unilabelAccuracy += 1
    print unilabelAccuracy / len(testingLabels)

    #Only relabel
    print "Relabeling Accuracy"
    relabelAccuracy = 0.0
    for testingLabel in testingLabels:
        if testingLabel > 0:
            relabelAccuracy += 1
    print relabelAccuracy / len(testingLabels)

#mlml(50, 100)
#mlml2()
#mlml3()
