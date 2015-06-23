from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import sys, os, traceback

from random import sample, random, randint
from MDP import *
from SimpleMDP import *
from SimpleMDP2 import *
from logisticRegression import *
from svm import *
from dt import *
from nn import *
from rf import *
from perceptron import *
from mnb import *
from sgd import *

from uct import *
from uct2 import *
from utils import *
from execLoops import *
#from samplingMethods import *
from samplingMethodClasses import *

from math import sqrt, floor
from data.makedata import makeGaussianData, makeTriGaussianData, makeLogicalData, makeUniformData
from data.makedataClasses import uniformData, gaussianData, realData
from ordereddict import OrderedDict

import cProfile, pstats, StringIO


pr = cProfile.Profile()
pr.enable()

#ThreadNumber is a string
threadNumber = None
if len(sys.argv) > 1:
    threadNumber = sys.argv[1]

#budget = 350
#let's assume all workers and difficulties are the same for now
# Accuracies: 55: 3.3, 65: 1.7, 75: 1.0, 85: 0.5, 95: 0.15
gamma = 1.0
#gamma = 0.0
d = 0.5

print "Accuracy of the workers:"
print 0.5 * (1.0 + (1.0 - d) ** gamma)

#Read the data
#f = open('data/data_banknote_authentication.txt', 'r') # 4 features
#f = open('data/breast-cancer-wisconsin-formatted.data', 'r') # 9 features
#f = open('data/iris.data', 'r') #4 features
#f = open('data/eegeyestate.data', 'r') #14 features 
#f = open('data/seismicbumps.data', 'r') #18 features
#f = open('data/wdbc-formatted.data', 'r') #30 features
#f = open('data/sonar.data', 'r') #60 features
#f = open('data/hv.data', 'r') #100 features
#f = open('data/hvnoise.data', 'r') #100 features
#f = open('data/ad-formatted.data', 'r') #1558 features
#f = open('data/arrhythmia2.data', 'r') #279 features
#f = open('data/musk.data', 'r') #166 features
#f = open('data/madelon.data', 'r') #500 features
#f = open('data/gisette.data', 'r') #5000 features
#f = open('data/adrandom.data', 'r') #6558 features
#f = open('data/adgisette.data', 'r') #6558 features
#f = open('data/farm-ads-formatted.data', 'r') #54877 features
#f = open('data/winequality-5binary-red.data', 'r') #11 features, 2 classes
#f = open('data/winequality-5binary-white.data', 'r') #11 features, 2 classes
#f = open('data/abalone_9binary.data', 'r') #8 features, 2 classes
#f = open('data/processed.cleveland.data', 'r') #13 features, 2 classes
#f = open('data/forestfires_0binary.data', 'r') #12 features, 2 classes

#f = open('data/optdigits.data', 'r') #64 features, 10 classes
#f = open('data/pendigits.data', 'r') #16 attributes, 10 classes
#f = open('data/winequality-white.data', 'r') #11 attributes, 11 classes
#f = open('data/cmc.data', 'r') #9 attributes, 3 classes
#f = open('data/sensor_readings_24.data', 'r') #24 attributes, 4 classes

realDataName = 'g7R'
numFeatures = 90
numClasses = 2
budget = 1000
#budget = int(sum(1 for line in f) / 2.0)
numberOfSimulations = 10


instances = []
classes = []
numberTrue = 0


#samplingStrategies = [passive(), uncertaintySampling(), 
#                      uncertaintySamplingAlpha(0.0),
#                      uncertaintySamplingAlpha(0.1),
#                      uncertaintySamplingAlpha(0.5),
#                      uncertaintySamplingAlpha(0.9)]


"""
samplingStrategies = [uncertaintySamplingAlpha(0.1),
                      uncertaintySamplingAlpha(0.3),
                      uncertaintySamplingAlpha(0.5),
                      uncertaintySamplingAlpha(0.7),
                      uncertaintySamplingAlpha(0.9)]
"""

#samplingStrategies = [uncertaintySamplingAlpha(0.9)]

#samplingStrategies = [validationSampling()]
#samplingStrategies = [cvSampling()]
#samplingStrategies = [cvSamplingBatch()]
#samplingStrategies = [passive()]
#samplingStrategies = [impactSampling()]
#samplingStrategies = [impactSamplingAll()]
#samplingStrategies = [impactSampling(numBootstrapSamples = 10)]
#samplingStrategies = [impactSampling(optimism=True)]
#samplingStrategies = [impactSampling(optimism=True, symmetric = True)]
#samplingStrategies = [impactSampling(optimism=True, pseudolookahead = True)]
#samplingStrategies = [impactSampling(pseudolookahead = True, symmetric = True)]
samplingStrategies = [impactSampling(pseudolookahead = True, optimism = True,
                                     symmetric = True)]

#samplingStrategies = [uncertaintySampling()]
#samplingStrategies = [uncertaintySamplingRelabel(3),
#                      uncertaintySamplingRelabel(5),
#                      uncertaintySamplingRelabel(7),
#                      uncertaintySamplingRelabel(9),
#                      uncertaintySamplingRelabel(11),]
#samplingStrategies = [passive()]
#samplingStrategies = [impactSampling(),
#                      uncertaintySampling()]

#samplingStrategies = [bayesianUncertaintySampling()]


"""
samplingStrategies = [impactSampling(),
                      impactSampling(optimism = True),
                      impactSampling(pseudolookahead = True),
                      impactSampling(optimism=True, 
                                     pseudolookahead=True)]

"""


"""
neighbor = impactSampling(optimism=True, pseudolookahead=True,
                          strategies = 
                          [uncertaintySampling(),
                           uncertaintySamplingLabeled(),
                           uncertaintySamplingAlpha(0.1),
                           uncertaintySamplingAlpha(0.3),
                           uncertaintySamplingAlpha(0.5),
                           uncertaintySamplingAlpha(0.7),
                           uncertaintySamplingAlpha(0.9)])
"""

"""
random = randomSampling(strategies = 
                          [uncertaintySampling(),
                           uncertaintySamplingLabeled(),
                           uncertaintySamplingAlpha(0.1),
                           uncertaintySamplingAlpha(0.3),
                           uncertaintySamplingAlpha(0.5),
                           uncertaintySamplingAlpha(0.7),
                           uncertaintySamplingAlpha(0.9)])
"""

"""
samplingStrategies = [impactSampling(),
                      impactSampling(optimism = True),
                      impactSampling(pseudolookahead = True),
                      impactSampling(optimism=True, 
                                     pseudolookahead=True),
                      uncertaintySampling(),
                      uncertaintySamplingAlpha(0.5),
                      passive(),
                      neighbor]
"""


#samplingStrategies = [neighbor]
#samplingStrategies = [uncertaintySamplingRelabel(5)]
#samplingStrategies = [uncertaintySamplingAlpha(0.9)]



numRelabels = 4
activeLearningExamples = 50


#dataGenerator = uniformData(budget*2, numFeatures)
dataGenerator = gaussianData(budget*2, numFeatures)
#dataGenerator = realData(budget * 2, numFeatures, f, realDataName)
    
files = []
statfiles = []
folderName = dataGenerator.getName()

for samplingStrategy in samplingStrategies:
    basefilename = 'outputs/%s/%s-f%d-lr-g%.1f-%d-%d' % (
        folderName,samplingStrategy.getName(), 
        numFeatures, gamma, budget, numberOfSimulations)
    if threadNumber == None:
        statfilename = basefilename + '-stats'
        impactstatfilename = basefilename + '-impactStats'
    else:
        statfilename = basefilename + '-stats-%s' % threadNumber
        impactstatfilename = basefilename + '-impactStats-%s' % threadNumber
        basefilename += '-%s' % threadNumber

    files.append(open(basefilename, 'w'))
    statfiles.append(open(statfilename, 'w'))
    #if samplingStrategy.__class__.__name__ == 'impactSamplingEMMedium':
    samplingStrategy.logFile = open(impactstatfilename, 'w')
        

activeLearningScores = []
relearningScores1 = []
relearningScores3 = []
relearningScores5 = []
relearningScores7 = []
allScores = [[] for file in files]
allStats = [[] for file in files]

activeLearningSkips = []
relearningSkips1 = []
relearningSkips3 = []
relearningSkips5 = []
relearningSkips7 = []

activeLearningFScores = []
relearningFScores1 = []
relearningFScores3 = []
relearningFScores5 = []
relearningFScores7 = []

for numSim in range(0, numberOfSimulations):

                    
    dataGenerator.generateData()

    trainingTasks = dataGenerator.trainingTasks
    trainingTaskClasses= dataGenerator.trainingTaskClasses
    trainingTaskDifficulties = dataGenerator.trainingTaskDifficulties
    validationTasks = dataGenerator.validationTasks
    validationTaskClasses = dataGenerator.validationTaskClasses
    testingTasks = dataGenerator.testingTasks
    testingTaskClasses = dataGenerator.testingTaskClasses

    print "Simulation Number:"
    print numSim


    classifier = LRWrapper()
    #classifier = SVMWrapper()
    #classifier = DTWrapper()
    #classifier = NNWrapper()
    #classifier = RFWrapper()
    #classifier = PerceptronWrapper()
    #classifier = MNBWrapper()
    #classifier = SGDWrapper()



    #state = mdp.getInitialState()
    #state5 = mdp5.getInitialState()

    #state = [(0,0) for i in range(len(trainingTasks))]
    #state = [[0 for i in range(numClasses)] for j in range(len(trainingTasks))]
    #Let's seed the classifier with some labels
    #state = {-1 : budget}
    state = OrderedDict([(-1, budget)])
    #state5 = deepcopy(state)
    #state.append(budget)
    #state5.append(-1)
    #state5.append(budget)

    #print state

    #baselineState5 = deepcopy(state5)
    #baselineState = deepcopy(state)
    #baselineStateAL = deepcopy(state)

    lam = 1.0
    #lam = 10000.0

    print "Relearning"


    for (i, samplingStrategy) in zip(range(len(files)),samplingStrategies):
        dataGenerator.generateDuplicateData()
        strategyFile = files[i]
        statFile = statfiles[i]

        success = False
        numSkips = 0.0
        while not success:
            try:
                #classifier = LRWrapper(lam)
                (stats, accuracies) = learn(
                    1, deepcopy(state), dataGenerator,
                    samplingStrategy,
                    gamma, budget, 
                    classifier, strategyFile, statFile, 12, numClasses)
                success = True
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print exc_type
                print exc_value
                print exc_traceback
                traceback.print_exception(exc_type, exc_value, exc_traceback)
                print e
                read_input()
                numSkips += 1.0
        #print accuracies
        strategyFile.flush()
        statFile.flush()
        allScores[i].append(accuracies[0][0])
        allStats[i].append(stats)

    for samplingStrategy in samplingStrategies:
        samplingStrategy.reinit()

    """
    directMDP(deepcopy(baselineState5), trainingtasks, classifier, f8)

    
    
    ###########################################################################
    # MDP With Validation rewards
    ###########################################################################
    
    print "Doing the cheating version of our method"
    state = deepcopy(baselineState)
    while state[-1] > 0:
        #print "HERE"
        nextAction = planner2.run(state)
        #nextAction = planner.getGreedyAction(state)
        print nextAction
        newState = deepcopy(state)

        activeTasks = []
        activeTaskIndices = []
        relearnTasks = []
        relearnTaskIndices = []
        for (i, task, (trues, falses)) in zip(range(len(trainingTasks)), 
                                              trainingTasks, state[0:-1]):
            if trues == 0 and falses == 0:
                activeTasks.append(task)
                activeTaskIndices.append(i)
            else:
                relearnTasks.append(task)
                relearnTaskIndices.append(i)

        retrain(trainingTasks, state[0:-1], classifier)

        #If active learning
        if nextAction == 0:
            print classifier.score(testingTasks, testingTaskClasses)
            (probs, index) = classifier.getMostUncertainTask( 
                activeTasks,
                activeTaskIndices)
            (trues, falses) = state[index]

            print "INDEX"
            print index
        else:
            index = getRelabelingPolicy(relearnTaskIndices, state,
                                        (1.0 / 2.0)*(1.0+((1.0 - d) ** gamma)))
            (trues, falses) = state[index]
            print "INDEX"
            print index


        if simLabel(trainingTaskDifficulties[index], gamma, 
                    trainingTaskClasses[index]) == 1:
            newState[index] = (trues+1, falses)
        else:
            newState[index] = (trues, falses+1)
        newState[-1] -= 1

        print newState[index]
        state = newState
        retrain(trainingTasks, state[0:-1], classifier)
        print classifier.score(testingTasks, testingTaskClasses)
        f5.write('%f\t' % classifier.score(testingTasks, testingTaskClasses))
        f5a.write('%f\t' % classifier.fscore(testingTasks, testingTaskClasses))


    f5.write('\n')
    f5a.write('\n')

    retrain(trainingTasks, state[0:-1], classifier)
    print state[0:-1]
    print classifier.score(testingTasks, testingTaskClasses)
    
    """
    """
    ###########################################################################
    # Find and Fix
    ###########################################################################
    
    print "Doing find and fix"
    state = deepcopy(baselineState)
    previousScore = 0.001

    while state[-1] > 0:
        newState = deepcopy(state)

        activeTasks = []
        activeTaskIndices = []
        relearnedTasks = []
        relearnedTaskIndices = []
        relearnedTaskClasses = []
        incorrectTasks = []
        incorrectTaskIndices = []
        votedTasks = []
        votedTaskClasses = []
        numTrainingTasks = 0.0
        for (i, task, (trues, falses)) in zip(range(len(trainingTasks)), 
                                              trainingTasks, state[0:-1]):
            if trues == 0 and falses == 0:
                activeTasks.append(task)
                activeTaskIndices.append(i)
            else:
                if not trues == falses:
                    predictedClass = 1 if trues > falses else 0
                    numTrainingTasks += 1.0
                    votedTasks.append(task)
                    votedTaskClasses.append(predictedClass)
                    if trues + falses > 1:
                        relearnedTaskClasses.append(1 if trues > falses else 0)
                        relearnedTaskIndices.append(i)
                        relearnedTasks.append(task)
                    if not classifier.predict(task) == predictedClass:
                        incorrectTasks.append(task)
                        incorrectTaskIndices.append(i)
                

        retrain(trainingTasks, state[0:-1], classifier)

        print classifier.score(testingTasks, testingTaskClasses)
        if relearnedTasks == []:
            beforeScore = 0.01
        else:
            beforeScore = classifier.score(relearnedTasks, relearnedTaskClasses)
            beforeScore += classifier.score(votedTasks, votedTaskClasses)
        print beforeScore

        (probs, index) = classifier.getMostUncertainTask( 
            activeTasks,
            activeTaskIndices)
        (trues, falses) = state[index]
        
        print relearnedTasks

        
        if relearnedTasks == []:
            score1 = 0
            score2 = 0
        else:
            state[index] = (trues+1, falses)
            retrain(trainingTasks, state[0:-1], classifier)
            score1 = classifier.score(relearnedTasks, relearnedTaskClasses)
            score1 += classifier.score(votedTasks, votedTaskClasses)
            state[index] = (trues, falses+1)
            retrain(trainingTasks, state[0:-1], classifier)
            score2 = classifier.score(relearnedTasks, relearnedTaskClasses)
            score2 += classifier.score(votedTasks, votedTaskClasses)

        afterScore = max(score1, score2)

        #suppose we did active learning. Then look at how well learning
        # the new point predicts the relearned points. If it does worse,
        # then maybe we need to relabel some points. If it does better
        # then we can learn that new point.

        #Basically, we relearn when active learning won't do well.
        print "Scores:"
        print score1
        print score2
        #if afterScore < previousScore:
        #    nextAction = 1
        #else:
        #    nextAction = 0
        print "THE SCORES:"
        print beforeScore
        print afterScore

        if afterScore < beforeScore:
            nextAction = 1
        else:
            nextAction = 0
        #previousScore = max(score1, score2)

        print "Next Action"
        print nextAction
        if nextAction == 1: #If relabeling:
            #get the things the classifier is getting wrong
            index = sample(incorrectTaskIndices, 1)[0]
            for k in range(numRelabels):
                (trues, falses) = newState[index]
                if simLabel(trainingTaskDifficulties[index], gamma, 
                            trainingTaskClasses[index]) == 1:
                    newState[index] = (trues+1, falses)
                else:
                    newState[index] = (trues, falses+1)
                
                newState[-1] -= 1
                state = newState
                retrain(trainingTasks, state[0:-1], classifier)
                print "AFTER UNCERTAINTY: Lower is better"
                print classifier.getTotalUncertainty(testingTasks)
                print classifier.score(testingTasks, testingTaskClasses)
                f13.write('%f\t' % classifier.score(testingTasks, testingTaskClasses))
                f13a.write('%f\t' % classifier.fscore(testingTasks, testingTaskClasses))

        else: #if active learning
            if simLabel(trainingTaskDifficulties[index], gamma, 
                    trainingTaskClasses[index]) == 1:
                newState[index] = (trues+1, falses)
            else:
                newState[index] = (trues, falses+1)
            newState[-1] -= 1

            state = newState
            retrain(trainingTasks, state[0:-1], classifier)
            print classifier.score(testingTasks, testingTaskClasses)
            f13.write('%f\t' % classifier.score(testingTasks, testingTaskClasses))
            f13a.write('%f\t' % classifier.fscore(testingTasks, testingTaskClasses))


    f13.write('\n')
    f13a.write('\n')

    retrain(trainingTasks, state[0:-1], classifier)
    print state[0:-1]
    print classifier.score(testingTasks, testingTaskClasses)
    """
    """
    ############################################################################
    ###########################################################################
    
    print "Doing UCT-MAX with validation rewards" 
    state = deepcopy(baselineState)
    while state[-1] > 0:
        #print "HERE"
        nextAction = planner4.run(state)
        #nextAction = planner.getGreedyAction(state)
        print nextAction
        newState = deepcopy(state)

        activeTasks = []
        activeTaskIndices = []
        relearnTasks = []
        relearnTaskIndices = []
        for (i, task, (trues, falses)) in zip(range(len(trainingTasks)), 
                                              trainingTasks, state[0:-1]):
            if trues == 0 and falses == 0:
                activeTasks.append(task)
                activeTaskIndices.append(i)
            else:
                relearnTasks.append(task)
                relearnTaskIndices.append(i)

        retrain(trainingTasks, state[0:-1], classifier)

        #If active learning
        if nextAction == 0:
            print classifier.score(testingTasks, testingTaskClasses)
            ((pFalse, pTrue), index) = classifier.getMostUncertainTask( 
                activeTasks,
                activeTaskIndices)
            (trues, falses) = state[index]

            print "INDEX"
            print index
        else:
            ((pFalse, pTrue), index) = classifier.getMostUncertainTask( 
                relearnTasks,
                relearnTaskIndices)
            (trues, falses) = state[index]
            print "INDEX"
            print index


        if simLabel(trainingTaskDifficulties[index], gamma, 
                    trainingTaskClasses[index]) == 1:
            newState[index] = (trues+1, falses)
        else:
            newState[index] = (trues, falses+1)
        newState[-1] -= 1

        print newState[index]
        state = newState
        retrain(trainingTasks, state[0:-1], classifier)
        print classifier.score(testingTasks, testingTaskClasses)
        f7.write('%f\t' % classifier.score(testingTasks, testingTaskClasses))

    f7.write('\n')
    retrain(trainingTasks, state[0:-1], classifier)
    print state[0:-1]
    print classifier.score(testingTasks, testingTaskClasses)

    ###########################################################################
    ###########################################################################

    
    state = deepcopy(baselineState)
    print "Doing the hacky version of our method"
    #While we still have a budget
    while state[-1] > 0:
        #print "HERE"
        (value0, value1) = mdp3.getHackyLookaheadValues(state)
        print value0
        print value1
        if value0 <= value1:
            nextAction = 0
        else:
            nextAction = 1

        #nextAction = planner.getGreedyAction(state)
        print nextAction
        newState = deepcopy(state)

        activeTasks = []
        activeTaskIndices = []
        relearnTasks = []
        relearnTaskIndices = []
        for (i, task, (trues, falses)) in zip(range(len(trainingTasks)), 
                                              trainingTasks, state[0:-1]):
            if trues == 0 and falses == 0:
                activeTasks.append(task)
                activeTaskIndices.append(i)
            else:
                relearnTasks.append(task)
                relearnTaskIndices.append(i)

        retrain(trainingTasks, state[0:-1], classifier)

        #If active learning
        if nextAction == 0:
            print classifier.score(testingTasks, testingTaskClasses)
            ((pFalse, pTrue), index) = classifier.getMostUncertainTask( 
                activeTasks,
                activeTaskIndices)
            (trues, falses) = state[index]

            print "INDEX"
            print index
        else:
            ((pFalse, pTrue), index) = classifier.getMostUncertainTask( 
                relearnTasks,
                relearnTaskIndices)
            (trues, falses) = state[index]
            print "INDEX"
            print index


        if simLabel(trainingTaskDifficulties[index], gamma, 
                    trainingTaskClasses[index]) == 1:
            newState[index] = (trues+1, falses)
        else:
            newState[index] = (trues, falses+1)
        newState[-1] -= 1
        state = newState
        retrain(trainingTasks, state[0:-1], classifier)
        print classifier.score(testingTasks, testingTaskClasses)
        f6.write('%f\t' % classifier.score(testingTasks, testingTaskClasses))

    f6.write('\n')
    retrain(trainingTasks, state[0:-1], classifier)
    print state[0:-1]
    print classifier.score(testingTasks, testingTaskClasses)

    ###########################################################################
    # Doing MDP with entropy rewards
    ###########################################################################

    state = deepcopy(baselineState)
    print "Doing our method"
    #While we still have a budget
    while state[-1] > 0:
        #print "HERE"
        nextAction = planner.run(state)
        #nextAction = planner.getGreedyAction(state)
        print nextAction
        newState = deepcopy(state)

        activeTasks = []
        activeTaskIndices = []
        relearnTasks = []
        relearnTaskIndices = []
        for (i, task, (trues, falses)) in zip(range(len(trainingTasks)), 
                                              trainingTasks, state[0:-1]):
            if trues == 0 and falses == 0:
                activeTasks.append(task)
                activeTaskIndices.append(i)
            else:
                relearnTasks.append(task)
                relearnTaskIndices.append(i)

        retrain(trainingTasks, state[0:-1], classifier)

        #If active learning
        if nextAction == 0:
            print classifier.score(testingTasks, testingTaskClasses)
            ((pFalse, pTrue), index) = classifier.getMostUncertainTask( 
                activeTasks,
                activeTaskIndices)
            (trues, falses) = state[index]

            print "INDEX"
            print index
        else:
            ((pFalse, pTrue), index) = classifier.getMostUncertainTask( 
                relearnTasks,
                relearnTaskIndices)
            (trues, falses) = state[index]
            print "INDEX"
            print index


        if simLabel(trainingTaskDifficulties[index], gamma, 
                    trainingTaskClasses[index]) == 1:
            newState[index] = (trues+1, falses)
        else:
            newState[index] = (trues, falses+1)
        newState[-1] -= 1
        state = newState
        retrain(trainingTasks, state[0:-1], classifier)
        print classifier.score(testingTasks, testingTaskClasses)
        f1.write('%f\t' % classifier.score(testingTasks, testingTaskClasses))

    f1.write('\n')
    retrain(trainingTasks, state[0:-1], classifier)
    print state[0:-1]
    print classifier.score(testingTasks, testingTaskClasses)

    
    ###########################################################################
    ###########################################################################

    print "Doing the random version of our method"
    state = deepcopy(baselineState)
    while state[-1] > 0:
        #print "HERE"
        nextAction = sample([0,1], 1)[0]
        print nextAction
        newState = deepcopy(state)

        activeTasks = []
        activeTaskIndices = []
        relearnTasks = []
        relearnTaskIndices = []
        for (i, task, (trues, falses)) in zip(range(len(trainingTasks)), 
                                              trainingTasks, state[0:-1]):
            if trues == 0 and falses == 0:
                activeTasks.append(task)
                activeTaskIndices.append(i)
            else:
                relearnTasks.append(task)
                relearnTaskIndices.append(i)

        retrain(trainingTasks, state[0:-1], classifier)

        #If active learning
        if nextAction == 0:
            print classifier.score(testingTasks, testingTaskClasses)
            ((pFalse, pTrue), index) = classifier.getMostUncertainTask( 
                activeTasks,
                activeTaskIndices)
            (trues, falses) = state[index]

            print "INDEX"
            print index
        else:
            ((pFalse, pTrue), index) = classifier.getMostUncertainTask( 
                relearnTasks,
                relearnTaskIndices)
            (trues, falses) = state[index]
            print "INDEX"
            print index


        if simLabel(trainingTaskDifficulties[index], gamma, 
                    trainingTaskClasses[index]) == 1:
            newState[index] = (trues+1, falses)
        else:
            newState[index] = (trues, falses+1)
        newState[-1] -= 1
        state = newState
        retrain(trainingTasks, state[0:-1], classifier)
        print classifier.score(testingTasks, testingTaskClasses)
        f4.write('%f\t' % classifier.score(testingTasks, testingTaskClasses))

    f4.write('\n')
    retrain(trainingTasks, state[0:-1], classifier)
    print state[0:-1]
    print classifier.score(testingTasks, testingTaskClasses)


    print "Doing Random Labeling"
    #Random labeling
    while baselineState[-1] > 0:
        #print "HERE"
        index = sample(range(0, len(trainingTasks)), 1)[0]

        newState = deepcopy(baselineState)

        if simLabel(trainingTaskDifficulties[index], gamma, 
                    trainingTaskClasses[index]) == 1:
            newState[index] = (trues+1, falses)
        else:
            newState[index] = (trues, falses+1)
        newState[-1] -= 1
        baselineState = newState
        retrain(trainingTasks, baselineState[0:-1], classifier)
        print classifier.score(testingTasks, testingTaskClasses)
        f2.write('%f\t' % classifier.score(testingTasks, testingTaskClasses))

    f2.write('\n')

    retrain(trainingTasks, baselineState[0:-1], classifier)
    print classifier.score(testingTasks, testingTaskClasses)

    
    """

    """
    ###########################################################################
    #
    ###########################################################################
    #Only Active Learning
    #Random labeling

    state = deepcopy(baselineState)
    print "Doing Active Learning Only"
    while state[-1] > 0:

        activeTasks = []
        activeTaskIndices = []

        for (i, task, (trues, falses)) in zip(range(len(trainingTasks)), 
                                              trainingTasks, 
                                              state[0:-1]):
            if trues == 0 and falses == 0:
                activeTasks.append(task)
                activeTaskIndices.append(i)

        retrain(trainingTasks, state[0:-1], classifier)
        #print activeTaskIndices
        #(probs, index) = classifier.getMostUncertainTask( 
        #    activeTasks,
        #    activeTaskIndices)
        index = sample(activeTaskIndices, 1)[0]
        (trues, falses) = state[index]


        if simLabel(trainingTaskDifficulties[index], gamma, 
                    trainingTaskClasses[index]) == 1:
            state[index] = (trues+1, falses)
        else:
            state[index] = (trues, falses+1)
        state[-1] -= 1
        #print "AFTER UNCERTAINTY: Lower is better"
        #print classifier.getTotalUncertainty(testingTasks)
        #print classifier.score(testingTasks, testingTaskClasses)
        f3.write('%f\t' % classifier.score(testingTasks, testingTaskClasses))
        f3a.write('%f\t' % classifier.fscore(testingTasks, testingTaskClasses))

    f3.write('\n')
    f3a.write('\n')

    #print classifier.score(testingTasks, testingTaskClasses)
    
    """
    """
    ###########################################################################
    # Zhao et al.
    ###########################################################################

    state = deepcopy(baselineState)
    print "Doing Zhao et al."
    while state[-1] > 0:
        #print "HERE"

        newState = deepcopy(state)

        activeTasks = []
        activeTaskIndices = []
        print "Heuristic"
        print mdp.getHeuristic(state, 0) 
        numRelabels = 3
        #print "UNCERTAINTY: Lower is better"
        #print classifier.getTotalUncertainty(testingTasks)
        
        for (i, task, (trues, falses)) in zip(range(len(trainingTasks)), 
                                              trainingTasks, 
                                              state[0:-1]):
            if trues + falses < numRelabels:
                activeTasks.append(task)
                activeTaskIndices.append(i)

        retrain(trainingTasks, state[0:-1], classifier)
        print "BEFORE UNCERTAINTY: Lower is better"
        print classifier.getTotalUncertainty(testingTasks)
        print classifier.score(testingTasks, testingTaskClasses)
        (probs, index) = classifier.getMostUncertainTask( 
            activeTasks,
            activeTaskIndices)
        print "INDEX"
        print index


        for k in range(numRelabels):
            (trues, falses) = newState[index]
            if simLabel(trainingTaskDifficulties[index], gamma, 
                        trainingTaskClasses[index]) == 1:
                newState[index] = (trues+1, falses)
            else:
                newState[index] = (trues, falses+1)
                
            newState[-1] -= 1
            state = newState

            retrain(trainingTasks, state[0:-1], classifier)
            print "AFTER UNCERTAINTY: Lower is better"
            print classifier.getTotalUncertainty(testingTasks)
            print classifier.score(testingTasks, testingTaskClasses)
            f12.write('%f\t' % classifier.score(testingTasks, testingTaskClasses))
            f12a.write('%f\t' % classifier.fscore(testingTasks, testingTaskClasses))



    f12.write('\n')
    f12a.write('\n')

    retrain(trainingTasks, state[0:-1], classifier)
    print classifier.score(testingTasks, testingTaskClasses)
    
    """
    """
    
    ##########################################################################
    # Do only relearning
    ##########################################################################
    print "Doing Relearning Only"
    state = deepcopy(baselineState)

    while state[-1] > 0:

        newState = deepcopy(state)

        activeTasks = []
        activeTaskIndices = []
        #print "Heuristic"
        #print mdp.getHeuristic(state, 0) 
        #print "UNCERTAINTY: Lower is better"
        #print classifier.getTotalUncertainty(testingTasks)
        for (i, task, (trues, falses)) in zip(range(len(trainingTasks)), 
                                              trainingTasks, 
                                              state[0:-1]):
            if trues > 0 or falses > 0:
                activeTasks.append(task)
                activeTaskIndices.append(i)

        retrain(trainingTasks, state[0:-1], classifier)

        #print "BEFORE UNCERTAINTY: Lower is better"
        #print classifier.getTotalUncertainty(testingTasks)
        #print classifier.score(testingTasks, testingTaskClasses)
        #(probs, index) = classifier.getMostUncertainTask( 
        #    activeTasks,
        #    activeTaskIndices)
        index = getRelabelingPolicy(activeTaskIndices, state,
                                     (1.0 / 2.0) * (1.0 + ((1.0 - d) ** gamma))
)
        (trues, falses) = state[index]


        numMoreLabels = 1
        for r in range(numMoreLabels):
            (trues, falses) = newState[index]
            if simLabel(trainingTaskDifficulties[index], gamma, 
                        trainingTaskClasses[index]) == 1:
                newState[index] = (trues+1, falses)
            else:
                newState[index] = (trues, falses+1)
            newState[-1] -= 1
        state = newState
        retrain(trainingTasks, state[0:-1], classifier)
        #print "AFTER UNCERTAINTY: Lower is better"
        #print classifier.getTotalUncertainty(testingTasks)
        #print classifier.score(testingTasks, testingTaskClasses)
        f9.write('%f\t' % classifier.score(testingTasks, testingTaskClasses))
        f9a.write('%f\t' % classifier.fscore(testingTasks, testingTaskClasses))
        f10.write('%f\t' % labelAccuracy(trainingTasks, state[0:-1],
                                         trainingTaskClasses))
    for (trues, falses) in state[0:-1]:
        #if trues == 0  and falses == 0:
        #    continue
        #print (trues, falses)
        f11.write('%f\t' % (trues+falses))
    f11.write('\n')
    f9.write('\n')
    f9a.write('\n')
    f10.write('\n')

    retrain(trainingTasks, state[0:-1], classifier)
    #print classifier.score(testingTasks, testingTaskClasses)
    
    """


    """
    ##########################################################################
    # Do Pac Learning
    ##########################################################################
    print "Doing PAC Learning"
    state = deepcopy(baselineState)

    activeLearningCount = 0

    while activeLearningCount < activeLearningExamples:

        newState = deepcopy(state)
            
        activeTasks = []
        activeTaskIndices = []
        
        for (i, task, (trues, falses)) in zip(range(len(trainingTasks)), 
                                              trainingTasks, 
                                              state[0:-1]):
            if trues == 0 and falses == 0:
                activeTasks.append(task)
                activeTaskIndices.append(i)

        retrain(trainingTasks, state[0:-1], classifier)

        (probs, index) = classifier.getMostUncertainTask( 
            activeTasks,
            activeTaskIndices)

        (trues, falses) = state[index]
        

        if simLabel(trainingTaskDifficulties[index], gamma, 
                    trainingTaskClasses[index]) == 1:
            newState[index] = (trues+1, falses)
        else:
            newState[index] = (trues, falses+1)
        newState[-1] -= 1
        state = newState
        retrain(trainingTasks, state[0:-1], classifier)
        #print "AFTER UNCERTAINTY: Lower is better"
        #print classifier.getTotalUncertainty(testingTasks)
        #print classifier.score(testingTasks, testingTaskClasses)
        f14.write('%f\t' % classifier.score(testingTasks, testingTaskClasses))
        f14a.write('%f\t' % classifier.fscore(testingTasks, testingTaskClasses))
        activeLearningCount += 1
        #print activeLearningCount
    while state[-1] > 0:
        newState = deepcopy(state)
            
        activeTasks = []
        activeTaskIndices = []
        
        for (i, task, (trues, falses)) in zip(range(len(trainingTasks)), 
                                              trainingTasks, 
                                              state[0:-1]):
            if trues > 0 or falses > 0:
                activeTasks.append(task)
                activeTaskIndices.append(i)

        retrain(trainingTasks, state[0:-1], classifier)

        index = getRelabelingPolicy(activeTaskIndices, state,
                                     (1.0 / 2.0) * (1.0 + ((1.0 - d) ** gamma))
)

        (trues, falses) = state[index]


        if simLabel(trainingTaskDifficulties[index], gamma, 
                    trainingTaskClasses[index]) == 1:
            newState[index] = (trues+1, falses)
        else:
            newState[index] = (trues, falses+1)
        newState[-1] -= 1
        state = newState
        retrain(trainingTasks, state[0:-1], classifier)
        #print "AFTER UNCERTAINTY: Lower is better"
        #print classifier.getTotalUncertainty(testingTasks)
        #print classifier.score(testingTasks, testingTaskClasses)
        f14.write('%f\t' % classifier.score(testingTasks, testingTaskClasses))
        f14a.write('%f\t' % classifier.fscore(testingTasks, testingTaskClasses))
    f14.write('\n')
    f14a.write('\n')

    retrain(trainingTasks, state[0:-1], classifier)
    #print classifier.score(testingTasks, testingTaskClasses)
    """

pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()

print "All Scores:"
for (i, samplingStrategy) in zip(range(len(samplingStrategies)),
                                 samplingStrategies):
    print samplingStrategy.getName()
    print np.average(allScores[i])
    print 1.96 * np.std(allScores[i]) / sqrt(len(allScores[i]))
    
print "All Stats:"
for (i, samplingStrategy) in zip(range(len(samplingStrategies)),
                                 samplingStrategies):
    print samplingStrategy
    print "number Examples"
    print np.average([j[0] for j in allStats[i]])
    print "number Examples Relabeled"
    print np.average([j[1] for j in allStats[i]])
    print "number Times Relabeled"
    print np.average([j[2] for j in allStats[i]])

    #print 1.96 * np.std(allScores[i]) / sqrt(len(allScores[i]))

print "Active Learning Score:"
print np.average(activeLearningScores)
print 1.96 * np.std(activeLearningScores) / sqrt(len(activeLearningScores))
print np.average(activeLearningSkips)
print "Relearning Score:"
#print np.average(relearningScores1)
#print 1.96 * np.std(relearningScores1) / sqrt(len(relearningScores1))

print np.average(relearningScores3)
print 1.96 * np.std(relearningScores3) / sqrt(len(relearningScores3))
print np.average(relearningSkips3)

print np.average(relearningScores5)
print 1.96 * np.std(relearningScores5) / sqrt(len(relearningScores5))
print np.average(relearningSkips5)

print np.average(relearningScores7)
print 1.96 * np.std(relearningScores7) / sqrt(len(relearningScores7))
print np.average(relearningSkips7)

"""
print "Fscores:"
print np.average(activeLearningFScores)
print 1.96 * np.std(activeLearningFScores) / sqrt(len(activeLearningFScores))
print "Relearning Score:"
print np.average(relearningFScores3)
print 1.96 * np.std(relearningFScores3) / sqrt(len(relearningFScores3))

print np.average(relearningFScores5)
print 1.96 * np.std(relearningFScores5) / sqrt(len(relearningFScores5))

print np.average(relearningFScores7)
print 1.96 * np.std(relearningFScores7) / sqrt(len(relearningFScores7))
"""
