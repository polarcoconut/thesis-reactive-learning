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

###########################################################################
# MDP With Direct rewards
###########################################################################    
def directMDP(state, trainingTasks, classifier, outputfile, interval):

    print "new reward function"

    while state[-1] > 0:
        print "HERE"
        nextAction = planner5.run(state)
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
        elif nextAction == 2: #If doing the exact same action as before
            index = state[-2]
            (trues, falses) = state[index]
            pTrue = trues / (trues + falses)
            pFalse = 1.0 - pTrue
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
        newState[-2] = index

        print newState[index]
        state = newState
        retrain(trainingTasks, state[0:-1], classifier)
        print classifier.score(testingTasks, testingTaskClasses)
        outputfile.write('%f\t' % classifier.score(testingTasks, testingTaskClasses))

    outputfile.write('\n')
    retrain(trainingTasks, state[0:-1], classifier)
    print state[0:-1]
    print classifier.score(testingTasks, testingTaskClasses)
