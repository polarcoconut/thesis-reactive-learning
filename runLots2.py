from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import cPickle as pickle
import sys,os

from random import sample, random, randint, shuffle
from MDP import *
from SimpleMDP import *
from SimpleMDP2 import *
#from logisticRegression import *
from svm import *
from dt import *
from nn import *
from rf import *

from uct import *
from uct2 import *
from utils import *
from execLoops import *

from data.makedata import makeGaussianData, makePairPlaneData, makeSmallMarginData
from math import sqrt, floor, ceil

#import matplotlib.pyplot as plt 
import cProfile, pstats, StringIO

#import weka.core.jvm as jvm

#jvm.start()

budget = 20
budgetInterval = 1
#workerAccuracies = [3.3, 1.7, 1.0, 0.5, 0.15]
#workerAccuracies = [3.3]
workerAccuracies = [1.0]
#workerAccuracies = [0.0]
d = 0.5
averageAccuracies = {}
means = {}
standardErrors = {}
ratios = {}
meanRatios = {}
ratioSE = {}
numSimulations = 10000
numFeatures = 2
numExamples = 25
numClasses = 2

activeLearningSkips = []

numExamples1 = []

classifier = LRWrapper()
#classifier = DTWrapper()
#classifier = SVMWrapper()
#classifier = RFWrapper()
#classifier = NNWrapper()
#classifier = PerceptronWrapper()

budgets = range(budgetInterval, budget+budgetInterval, budgetInterval)
for gamma in workerAccuracies:
    for b in budgets:
        averageAccuracies[(gamma,b)] = []
        means[(gamma, b)] = 0
        standardErrors[(gamma, b)] = 0

for simNum in range(numSimulations):
    print simNum
    print "Writing Data"
    #writedatafile = open('runLotsData/%d,%d.data' % (numFeatures,numExamples),
    #                     'w')
    (instances, classes) = makeGaussianData(numExamples, numFeatures, 0, 
                     noise=False, numClasses=2, skew=1.0, 
                     f = None, randomData=True,
                     writeToFile = False)

    #makeSmallMarginData(numExamples, numFeatures, 0, 
    #                    noise=False, numClasses=2, skew=1.0,
    #                    f = writedatafile, randomData=True)

    #makePairPlaneData(numExamples, numFeatures, 0, noise=False, numClasses=2,
    #                  skew = 1.0, f=writedatafile, randomData=True)

    #writedatafile.close()
    #readdatafile = open('runLotsData/%d,%d.data' % (numFeatures, numExamples),
    #                    'r')

    #readdatafile = open('data/gisette.data', 'r')
    #readdatafile = open('data/xiaogaussian.data', 'r')

    """
    print "reading data"
    instances = []
    classes = []
    numberTrue = 0
    for line in readdatafile:
        line = line.split(',')
        #print len(line)
        instances.append([float(line[i]) for i in range(numFeatures)])
        #if (int(line[numFeatures])) == 1:
        #    classes.append(1)
        #else:
        #    classes.append(0)
        classes.append(int(line[numFeatures]))
    """
    #readdatafile.close()
    
    #print len(instances)
    trainingTasks = []
    trainingTaskClasses = []
    trainingTaskDifficulties = []

    validationTasks = []
    validationTaskClasses = []
    
    testingTasks = []
    testingTaskClasses = []


    allSamples = zip(instances, classes)
    shuffle(allSamples)
    trainingSamples = allSamples[0:int(ceil(0.8 * len(allSamples)))]
    validationSamples = allSamples[int(ceil(0.8 * len(allSamples))):int(ceil(0.85 * len(allSamples)))]
    testingSamples = allSamples[int(ceil(0.85 * len(allSamples))):len(allSamples)]


    for (instance, c) in trainingSamples:
        trainingTasks.append(instance)
        trainingTaskClasses.append(c)
        trainingTaskDifficulties.append(d)
    for (instance, c) in validationSamples:
        validationTasks.append(instance)
        validationTaskClasses.append(c)
    for (instance, c) in testingSamples:
        testingTasks.append(instance)
        testingTaskClasses.append(c)

    state = [[0 for i in range(numClasses)] for j in range(len(trainingTasks))]
    state.append(budget)
    baselineState = deepcopy(state)

    print "beginning experiments"

    for gamma in workerAccuracies:
        #print "Active Learning"
        success = False
        numSkips = 0.0
        unilabelingAccuracies = []
        while not success:
            try:
                #classifier = LRWrapper(lam)
                (nExamplesUsed, accuracies) = learn(1, deepcopy(baselineState), 
                                        trainingTasks, 
                                        trainingTaskDifficulties, 
                                        trainingTaskClasses, 
                                        testingTasks,testingTaskClasses, 
                                        gamma, budget, 
                                        classifier, None, 12, numClasses,
                                        bayesOptimal = False,
                                        smartBudgeting = True,
                                        budgetInterval = budgetInterval)
                numExamples1.append(nExamplesUsed)
                for (i, b) in zip(range(len(budgets)), budgets):
                    #print accuracies
                    #print averageAccuracies
                    #print accuracies[i][0]
                    unilabelingAccuracies.append(accuracies[i][0])
                    averageAccuracies[(gamma, b)].append(accuracies[i][0])
                success = True
            except Exception as e:
                print "1"
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print e
                numSkips += 1.0
        activeLearningSkips.append(numSkips)


#Get the results and figure out the winners
for (i, gamma) in zip(range(len(workerAccuracies)), workerAccuracies):
    for (j, b) in zip(range(len(budgets)), budgets):
        means[(gamma, b)] = np.average(averageAccuracies[(gamma, b)])
        standardErrors[(gamma, b)] = (1.96 * np.std(
            averageAccuracies[(gamma, b)]))/sqrt(numSimulations)
            


print means
print standardErrors


pickle.dump(means, open(
        "results/runlotsAccuracies%d,%d,%d,%d" % 
        (numFeatures, numExamples, budget, budgetInterval),
        "wb"))
pickle.dump(standardErrors, open(
        "results/runlotsErrors%d,%d,%d,%d" % 
        (numFeatures, numExamples, budget, budgetInterval),
        "wb"))
