from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import cPickle as pickle
import sys,os

from random import sample, random, randint, shuffle
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

budget = 1180
budgetInterval = 1180
#workerAccuracies = [3.3, 1.7, 1.0, 0.5, 0.15]
#workerAccuracies = [8.0, 2.3, 1.3, 0.7, 0.32]
#workerAccuracies = [3.3]
workerAccuracies = [1.0]
#workerAccuracies = [0.0]
#workerAccuracies = [0.32]

d = 0.5
averageAccuracies = {}
means = {}
standardErrors = {}
ratios = {}
meanRatios = {}
ratioSE = {}
numSimulations = 1000
numFeatures = 1558
numExamples = 1180
numClasses = 2

activeLearningSkips = []
relearningSkips1 = []
relearningSkips3 = []
relearningSkips5 = []
relearningSkips7 = []

numExamples1 = []
numExamples3 = []
numExamples5 = []
numExamples7 = []


budgets = range(budgetInterval, budget+budgetInterval, budgetInterval)
for gamma in workerAccuracies:
    for b in budgets:
        averageAccuracies[(gamma,b)] = [[],[],[],[]]
        ratios[(gamma, b)] = [[],[],[]]
        meanRatios[(gamma, b)] = [0,0,0]
        ratioSE[(gamma, b)] = [0,0,0]
        means[(gamma, b)] = [0,0,0,0]
        standardErrors[(gamma, b)] = [0,0,0,0]

classifier = LRWrapper()
#classifier = DTWrapper()
#classifier = SVMWrapper()
#classifier = RFWrapper()
#classifier = NNWrapper()
#classifier = PerceptronWrapper()

for simNum in range(numSimulations):
    print simNum
    print "Writing Data"
    writedatafile = open('runLotsData/%d,%d.data' % (numFeatures,numExamples),
                         'w')
    makeGaussianData(numExamples, numFeatures, 0, 
                     noise=False, numClasses=2, skew=1.0, 
                     f = writedatafile, randomData=True,
                     writeToFile = True)

    #makeSmallMarginData(numExamples, numFeatures, 0, 
    #                    noise=False, numClasses=2, skew=1.0,
    #                    f = writedatafile, randomData=True)

    #makePairPlaneData(numExamples, numFeatures, 0, noise=False, numClasses=2,
    #                  skew = 1.0, f=writedatafile, randomData=True)

    writedatafile.close()
    readdatafile = open('runLotsData/%d,%d.data' % (numFeatures, numExamples),
                        'r')

    #readdatafile = open('data/gisette.data', 'r')
    #readdatafile = open('data/xiaogaussian.data', 'r')


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

    readdatafile.close()

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

    #print len(trainingSamples)

    #raise Exception

    #xiaostring = '%d'
    #for j in range(numFeatures):
    #    xiaostring += (' %d:' % (j+1))
    #    xiaostring += '%f'
    #xiaostring += '\n'
    #xfile = open('data/xiaogaussian.data', 'w')
    #yfile = open('data/xiaogisetteTest.data', 'w')

    for (instance, c) in trainingSamples:
        #temp = deepcopy(instance)
        #temp.insert(0,c)
        #xfile.write(xiaostring % tuple(temp))
        trainingTasks.append(instance)
        trainingTaskClasses.append(c)
        trainingTaskDifficulties.append(d)
    for (instance, c) in validationSamples:
        validationTasks.append(instance)
        validationTaskClasses.append(c)
    for (instance, c) in testingSamples:
        #temp = deepcopy(instance)
        #temp.insert(0,c)
        #xfile.write(xiaostring % tuple(temp))
        testingTasks.append(instance)
        testingTaskClasses.append(c)

    #xfile.close()
    #yfile.close()

    """
    for (instance, c) in zip(instances, classes):
        r = random()
        if r < 0.8:
            trainingTasks.append(instance)
            trainingTaskClasses.append(c)
            trainingTaskDifficulties.append(d) #Difficulty is constant
        elif r < 0.85:
            validationTasks.append(instance)
            validationTaskClasses.append(c)
        else:
            testingTasks.append(instance)
            testingTaskClasses.append(c)
    """
    state = [[0 for i in range(numClasses)] for j in range(len(trainingTasks))]
    state.append(budget)
    baselineState = deepcopy(state)

    lam = 1.0

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
                    averageAccuracies[(gamma, b)][0].append(accuracies[i][0])
                success = True
            except Exception as e:
                print "1"
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print e
                numSkips += 1.0
        activeLearningSkips.append(numSkips)

        success = False
        numSkips = 0.0
        while not success:
            try:
                #classifier = LRWrapper(lam * 3)
                (nExamplesUsed, accuracies) = learn(3, deepcopy(baselineState), 
                                        trainingTasks, 
                                        trainingTaskDifficulties, 
                                        trainingTaskClasses, 
                                        testingTasks,testingTaskClasses, 
                                        gamma, budget, 
                                        classifier, None, 12, numClasses, 
                                        bayesOptimal = False,
                                        smartBudgeting = True,
                                        budgetInterval = budgetInterval)
                numExamples3.append(nExamplesUsed)
                for (i, b) in zip(range(len(budgets)), budgets):
                    accuracy = accuracies[i][0]
                    averageAccuracies[(gamma, b)][1].append(accuracy)
                    #print accuracy
                    #print averageAccuracies[(gamma, b)][0][simNum]
                    #print accuracy / averageAccuracies[(gamma, b)][0][simNum]
                    ratios[(gamma, b)][0].append(
                        accuracy / averageAccuracies[(gamma,b)][0][simNum])
                #print averageAccuracies
                success = True
            except Exception as e:
                print "3"
                print e
                numSkips += 1.0
                
        #print accuracies
        relearningSkips3.append(numSkips)
        
        success = False
        numSkips = 0.0
        while not success:
            try:
                #classifier = LRWrapper(lam * 5)
                (nExamplesUsed, accuracies) = learn(5, deepcopy(baselineState), 
                                        trainingTasks, 
                                        trainingTaskDifficulties, 
                                        trainingTaskClasses, 
                                        testingTasks,testingTaskClasses, 
                                        gamma, budget, 
                                        classifier, None, 15, numClasses, 
                                        bayesOptimal = False,
                                        smartBudgeting = True,
                                        budgetInterval = budgetInterval)
                numExamples5.append(nExamplesUsed)
                for (i, b) in zip(range(len(budgets)), budgets):
                    accuracy = accuracies[i][0]
                    averageAccuracies[(gamma, b)][2].append(accuracy)
                    ratios[(gamma, b)][1].append(
                        accuracy / averageAccuracies[(gamma, b)][0][simNum])
                success = True
            except Exception as e:
                print "5"
                print e
                numSkips += 1.0
        relearningSkips5.append(numSkips)

        success = False
        numSkips = 0.0
        while not success:
            try:
                #classifier = LRWrapper(lam * 7)
                (nExamplesUsed, accuracies) = learn(7, deepcopy(baselineState), 
                                        trainingTasks, 
                                        trainingTaskDifficulties, 
                                        trainingTaskClasses, 
                                        testingTasks,testingTaskClasses, 
                                        gamma, budget, 
                                        classifier, None, 14, numClasses, 
                                        bayesOptimal = False,
                                        smartBudgeting = True,
                                        budgetInterval = budgetInterval)
                numExamples7.append(nExamplesUsed)
                for (i, b) in zip(range(len(budgets)), budgets):
                    accuracy = accuracies[i][0]
                    averageAccuracies[(gamma, b)][3].append(accuracy)
                    ratios[(gamma, b)][2].append(
                        accuracy / averageAccuracies[(gamma, b)][0][simNum])
                success = True
            except Exception as e:
                print "7"
                print e
                numSkips += 1.0
        relearningSkips7.append(numSkips)
        



#Get the results and figure out the winners
bestStrategies = np.zeros((len(workerAccuracies), len(budgets)))
for (i, gamma) in zip(range(len(workerAccuracies)), workerAccuracies):
    for (j, b) in zip(range(len(budgets)), budgets):
        means[(gamma, b)][0] = np.average(
                averageAccuracies[(gamma, b)][0])
        means[(gamma, b)][1] = np.average(
                averageAccuracies[(gamma, b)][1])
        means[(gamma, b)][2] = np.average(
                averageAccuracies[(gamma, b)][2])
        means[(gamma, b)][3] = np.average(
                averageAccuracies[(gamma, b)][3])

        meanRatios[(gamma, b)][0] = np.average(ratios[(gamma, b)][0])
        meanRatios[(gamma, b)][1] = np.average(ratios[(gamma, b)][1])
        meanRatios[(gamma, b)][2] = np.average(ratios[(gamma, b)][2])



        ratioSE[(gamma, b)][0] = (1.96 * np.std(
            ratios[(gamma, b)][0]))/sqrt(numSimulations)
        ratioSE[(gamma, b)][1] = (1.96 * np.std(
            ratios[(gamma, b)][1]))/sqrt(numSimulations)
        ratioSE[(gamma, b)][2] = (1.96 * np.std(
            ratios[(gamma, b)][2]))/sqrt(numSimulations)

        #print ratios[(gamma, b)]

        standardErrors[(gamma, b)][0] = (1.96 * np.std(
            averageAccuracies[(gamma, b)][0]))/sqrt(numSimulations)
        standardErrors[(gamma, b)][1] = (1.96 * np.std(
            averageAccuracies[(gamma, b)][1]))/sqrt(numSimulations)
        standardErrors[(gamma, b)][2] = (1.96 * np.std(
            averageAccuracies[(gamma, b)][2]))/sqrt(numSimulations)
        standardErrors[(gamma, b)][3] = (1.96 * np.std(
            averageAccuracies[(gamma, b)][3]))/sqrt(numSimulations)
            
        #print averageAccuracies[(gamma, b)]

        #averageAccuracies[(gamma, b)][0] /= numSimulations
        #averageAccuracies[(gamma, b)][1] /= numSimulations
        #averageAccuracies[(gamma, b)][2] /= numSimulations
        #averageAccuracies[(gamma, b)][3] /= numSimulations
        bestStrategies[i,j] = np.argmax(means[(gamma, b)])

#print averageAccuracies
print bestStrategies
print means
print standardErrors
print meanRatios
print ratioSE

print np.average(numExamples1)
print np.average(numExamples3)
print np.average(numExamples5)
print np.average(numExamples7)

pickle.dump(means, open(
        "results/runlotsAccuracies%d,%d,%d,%d" % 
        (numFeatures, numExamples, budget, budgetInterval),
        "wb"))
pickle.dump(standardErrors, open(
        "results/runlotsErrors%d,%d,%d,%d" % 
        (numFeatures, numExamples, budget, budgetInterval),
        "wb"))
pickle.dump(bestStrategies, open(
        "results/runlotsStrategies%d,%d,%d,%d" % 
        (numFeatures, numExamples, budget, budgetInterval), 
        "wb"))

"""

fig, ax = plt.subplots()
ax.matshow(bestStrategies, cmap = plt.cm.gray)
#ax.set_xticklabels(budgets)
plt.xticks(range(len(budgets)), budgets)
ax.xaxis.set_label_position('top')
ax.set_xlabel("Budget")
#ax.set_yticklabels([55, 65, 75, 85, 95])
plt.yticks(range(len(workerAccuracies)), [55,65,75,85,95])
ax.set_ylabel("Worker Accuracy")
plt.show()

"""
