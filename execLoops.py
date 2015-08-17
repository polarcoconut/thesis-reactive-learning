from random import sample, shuffle
from utils import *
#import matplotlib.pyplot as plt
from logisticRegression import *
#from logisticRegressionWEKA import *
from svm import *
from dt import *
from nn import *
from rf import *
from perceptron import *
import sys
from copy import deepcopy
from samplingMethods import *
import pickle

##########################################################################
# uses an active learning strategy to learn a classifier
# metric can be 'acc' or 'fscore'
##########################################################################
def learn(numRelabels, state, dataGenerator,
          samplingStrategy,
          gamma, budget, 
          classifier, 
          outputfile,
          statfile,
          interval,
          numClasses, metric = 'acc',
          bayesOptimal = False, smartBudgeting = False,
          budgetInterval = None):

    if budgetInterval == None:
        budgetInterval = budget

    accuracy = (1.0 / 2.0)*(1.0+((1.0 - 0.5) ** gamma))

    outputString = ""
    statOutputString = ""
    activeTasks = []
    activeTaskIndices = []


    accuracies = []


    #activeTaskIndices = range(len(trainingTasks))
    #shuffle(activeTaskIndices)
    #while state[-1] > 0:
    numExamples = 0.0
    totalSpentSoFar = 0.0

    goldLabels = []

    sampledIndices = []
    #a hack for the first point when the budgetinterval is 1
    firstPointManaged = False
    
    #for idx in range(len(deepcopy(activeTaskIndices))):
    idx = -1
    while True:
        #index = samplingStrategy(activeTaskIndices)
        #index = passive(activeTaskIndices)
        #Active Learning
        print "CURRENT INDEX"
        print idx
        idx += 1
        if idx < 61:
            nextTask = passive(dataGenerator, state,
                            classifier, accuracy)
        else:
            nextTask = samplingStrategy.sample(dataGenerator,
                                            state, classifier, accuracy)

        #print nextTask
        if nextTask not in state:
            #print "REMOVING"
            sampledIndices.append('a')
            #print len(dataGenerator.trainingTasks)
            #dataGenerator.trainingTasks.remove(nextTask)
            #print len(dataGenerator.trainingTasks)
            state[nextTask] = [0,0]
        else:
            #print "NOT REMOVING"
            sampledIndices.append('r')

        if nextTask in dataGenerator.trainingTasks:
            dataGenerator.trainingTasks = filter(
                lambda a: a != nextTask, dataGenerator.trainingTasks)
            #dataGenerator.trainingTasks.remove(nextTask)
        nextClass = dataGenerator.trainingTaskClasses[nextTask]
        if isinstance(nextClass, list):
            nextClass  = sample(nextClass, 1)[0]

        dataGenerator.replenish()

        #if index in activeTaskIndices:
        #    activeTaskIndices.remove(index)
            
        #print len(activeTaskIndices)
        #print numExamples
        #print state[-1]
        #print numRelabels
        #print state

        if state[-1] <= 0:
            break
        numExamples += 1
        #index = sample(activeTaskIndices, 1)[0]
        #activeTaskIndices.remove(index)

        """
        if sum(state[index]) > 0:
            numRelabels = 2
        else:
            numRelabels = 1
        """

        #(trues, falses) = state[index]
        for r in range(numRelabels):
            #(trues, falses) = state[index]
            incorrectLabels = [i for i in range(numClasses)]
            incorrectLabels.remove(nextClass)
            workerLabel = simLabel(0.5, gamma, nextClass, incorrectLabels)
            state[nextTask][workerLabel] += 1
            state[-1] -= 1

            #print state[-1]
            totalSpentSoFar += 1
            if totalSpentSoFar % budgetInterval == 0:
                if not firstPointManaged and budgetInterval == 1:
                    accuracies.append((0.5, 0.5))
                    firstPointManaged = True
                    continue
                #classifier = LRWrapper(1.0 * (1.0 * budget / numExamples))
                #classifier = LRWrapper(100000000.0)
                #classifier = SVMWrapper(1.0 * (1.0 * budget / numExamples))
                #classifier = DTWrapper()
                #classifier = RFWrapper()
                #classifier = NNWrapper()
                #classifier = PerceptronWrapper()

                classifier.C = 1.0 * (1.0 * budget / numExamples)
                #print "RETRAINING"
                #print totalSpentSoFar
                retrain(state, classifier, 
                        bayesOptimal, accuracy)
                #print len(testingTasks)
                #print classifier.predict(testingTasks[0:10])
                #print testingTaskClasses[0:10]
                #print classifier.score(testingTasks, testingTaskClasses)
                #print classifier.getParams()
                accuracies.append(
                    (classifier.score(dataGenerator.testingTasks, 
                                      dataGenerator.testingTaskClasses),
                     classifier.fscore(dataGenerator.testingTasks, 
                                       dataGenerator.testingTaskClasses)))
                print accuracies
                #print accuracies
            if smartBudgeting:
                if state[nextTask][workerLabel] > int(numRelabels / 2):
                    break
            if state[-1] <= 0:
                #print "TERMINATING"
                #print state[-1]
                break

        #print state[-1]

        #print budget-state[-1]
        
        if (samplingStrategy.validate() and 
            (budget-state[-1]) > 50):
            retrain(state, classifier, True, accuracy)
            samplingStrategy.setValidation(classifier.score(
                dataGenerator.validationTasks, 
                dataGenerator.validationTaskClasses))
        
        if ((budget - state[-1]) % interval == 0 and 
            (budget-state[-1]) > 50):
            #print "HERE"
            #print state
            retrain(state, classifier, True, accuracy)
            #pickle.dump(computeStats(state[0:-1]), statfile)
            if metric == 'acc':
                outputString+= ("%f\t"% classifier.score(
                    dataGenerator.testingTasks, 
                    dataGenerator.testingTaskClasses))
            elif metric == 'fscore':
                outputString+= ("%f\t"% classifier.fscore(
                    dataGenerator.testingTasks, 
                    dataGenerator.testingTaskClasses))
            else:
                print "Unknown Metric Specified"
                raise Exception('Unknown Metric Specified')
            statOutputString+= ("%f,%f\t"% computeStats(state)[1:])
        
    outputString += "\n"
    outputfile.write(outputString)
    statOutputString += "\n"
    statfile.write(statOutputString)
    #pickle.dump("END", statfile)

    #print trainingTasks
    #print state[0:-1]
    #print "RETRAINING"
    accuracy = (1.0 / 2.0)*(1.0+((1.0 - 0.5) ** gamma))

    classifier.C = 1.0 * (1.0 * budget / numExamples)
    #classifier = LRWrapper(0.01 * (1.0 * budget / numExamples))
    #classifier = LRWrapper(1.0 * (1.0 * budget / numExamples))
    #classifier = LRWrapper(100000000.0)
    #print budget
    #print numExamples
    #print 1.0 * (1.0 * budget / numExamples)
    #classifier = DTWrapper()
    #classifier = SVMWrapper(1.0 * (1.0 * budget / numExamples))
    #classifier = RFWrapper()
    #classifier = NNWrapper()
    #classifier = PerceptronWrapper()

    retrain(state, classifier, bayesOptimal, accuracy)
    #print classifier.getParams()
    #print labelAccuracy(trainingTasks, state[0:-1], trainingTaskClasses)

    
    #return (classifier.score(testingTasks, testingTaskClasses),
    #        classifier.fscore(testingTasks, testingTaskClasses))
    #print "HUH"
    #print accuracies
    print sampledIndices

    (sampledIndices, numExamplesRelabeled, 
     numTimesRelabeled) = computeStats(state) 

    #print "HUH"
    #print accuracies
    print numExamplesRelabeled, numTimesRelabeled
    return ((numExamples, numExamplesRelabeled, numTimesRelabeled), 
            accuracies)
