from random import random
from copy import deepcopy
#from us import *
from utils import *

class SimpleMDP2:


    #each task is a feature vector
    #we consider true = 1 and false = 0
    def __init__(self, trainingTasks, testingTasks, classifier, budget,
                 priorTrue, validationTasks = None, RL = False):
        
        self.priorTrue = priorTrue
        self.priorFalse = 1.0 - priorTrue
        self.tasks = trainingTasks
        self.testingTasks = testingTasks
        self.numberOfTasks = len(trainingTasks)
        self.classifier = classifier
        self.budget = budget

        self.activeLearningIndices = {}
        self.reLearningIndices = {}

        self.validationTasks = validationTasks
        self.RL = RL

        d = 0.5
        g = 3.0
        self.acc = (1.0 / 2.0) * (1.0 + ((1.0 - d) ** g))

    #the state is a list of trues and falses and then the current budget
    def getInitialState(self):
        initialState = [(0,0) for i in range(self.numberOfTasks)]
        initialState.append(-1)
        initialState.append(self.budget)
        return initialState
    
    #0 means active learning, 1 means relearning, 2 means redo the previous
    #action
    def getActions(self, state):
        if state[-2] == -1:
            return [0,1]
        else:
            return [0,1,2]

    def isTerminal(self, state):
        if state[-1] <= 0:
            return True
        else:
            return False


    #In this MDP, rewards are awarded upon taking an action and 
    #arriving at the state. Rewards are
    # only awarded when the budget is exhausted
    def getReward(self, state):
        currentBudget = state[-1]
        if currentBudget > 0:
            return 0
        else:
            numTrainingExamples = 0.0
            numSureExamples = 0.0
            relearnExamples = []
            for (task, (trues, falses)) in zip(self.tasks, state[0:-2]):
                if trues == 0 and falses == 0:
                    continue
                else:
                    numTrainingExamples += 1.0
                    relearnExamples.append(task)
            
            return 0.01 * numTrainingExamples + self.getNumSureExamples(state)
        
    def getNumSureExamples(self, state):
        numSureExamples = 0.0
        for (trues, falses) in state[0:-2]:
            if trues == 0 and falses == 0:
                continue
            #Calculate entropy of the posterior
            acc = self.acc
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
            
            #print min(pTrue, pFalse)
            #if min(pTrue, pFalse) == 0.5:
            #    print (trues, falses)
            if min(pTrue, pFalse) < 0.30:
                numSureExamples += 1
            #if min(pTrue, pFalse) < 0.50:
            #    numSureExamples += 0.1
        
        #print "HERE"
        #print numSureExamples
        return numSureExamples
                                     
    def getHeuristic(self, state, action):
        return 0
        #print state
        currentClassifier = self.classifier()
        ((p1, s1), (p2, s2)) = self.T(state, action)
        retrain(self.tasks, s1[0:-2], currentClassifier)
        
        h = p1 * currentClassifier.getTotalUncertainty(self.testingTasks)
        retrain(self.tasks, s2[0:-2], currentClassifier)
        
        h += p2 * currentClassifier.getTotalUncertainty(self.testingTasks)
        
        h *= -1.0
                                     
        #print h
        #return h

    """
    def evalClassifier(self, state):
        currentClassifier = self.classifier()
        retrain(self.tasks, state[0:-1], currentClassifier)
        if self.validationTasks == None:
            return currentClassifier.getTotalUncertainty(self.testingTasks)
        else:
            return -1.0 * currentClassifier.score(self.validationTasks[0],
                                           self.validationTasks[1])
                    

    def getHackyLookaheadValues(self, state):
        ((pTrue, newState01), (pFalse, newState02)) = self.T(state, 0)
        ((pTrue, newState11), (pFalse, newState12)) = self.T(state, 1)

        score01 = self.evalClassifier(newState01)
        score02 = self.evalClassifier(newState02)
        print "Scores"
        print score01
        print score02
        score11 = self.evalClassifier(newState11)
        score12 = self.evalClassifier(newState12)
        return (min(score01, score02), (pTrue * score11) + (pFalse * score12))

    """
    #this transition function does not mutate the state
    def T(self, state, action):
        
        newState1 = deepcopy(state)
        newState2 = deepcopy(state)

        activeTasks = []
        activeTaskIndices = []
        relearnTasks = []
        relearnTaskIndices = []
        for (i, task, (trues, falses)) in zip(range(len(self.tasks)), 
                                              self.tasks, state[0:-2]):
            if trues == 0 and falses == 0:
                activeTasks.append(task)
                activeTaskIndices.append(i)
            else:
                relearnTasks.append(task)
                relearnTaskIndices.append(i)

        currentClassifier = self.classifier()
        retrain(self.tasks, state[0:-2], currentClassifier)

        #If active learning
        if action == 0:
            ((pFalse, pTrue), index) = currentClassifier.getMostUncertainTask( 
                activeTasks,
                activeTaskIndices)
            (trues, falses) = state[index]

        elif action == 2: #If doing the exact same action as before
            #print state

            index = state[-2]
            #print "INDEX"
            #print index
            (trues, falses) = state[index]
            pTrue = trues / (trues + falses)
            pFalse = 1.0 - pTrue
        else:
            index = getRelabelingPolicy(relearnTaskIndices, state, self.acc)
            (trues, falses) = state[index]
            #pTrue = (trues + 1.0) / (trues  + falses + 2.0)
            #pFalse = 1.0 - pTrue
            pTrue = trues / (trues + falses)
            pFalse = 1.0 - pTrue

        acc = self.acc
        if self.RL:
            #If we are told the worker accuracy
            if not acc == None:
                averageWorkerAccuracy = acc
            else:
                averageWorkerAccuracy = 0.0
                count = 0.0
                for (ts, fs) in state[0:-2]:
                    if ts == 0 and fs == 0:
                        continue
                    if ts > fs:
                        averageWorkerAccuracy += ts / (fs + ts)
                    else:
                        averageWorkerAccuracy += fs/ (ts + fs)
                    count += 1
                averageWorkerAccuracy /= count
            
            pNewState1 = ((pTrue * averageWorkerAccuracy) + 
                          (pFalse *  (1.0 - averageWorkerAccuracy)))
            pNewState2 = ((pTrue * (1.0 - averageWorkerAccuracy)) +
                          (pFalse * averageWorkerAccuracy))
        else:
            pNewState1 = pTrue
            pNewState2 = pFalse

        
        mostUncertainTask = self.tasks[index]
        
        newState1[index] = (trues+1, falses)
        newState2[index] = (trues, falses+1)
            
        #for now, each task costs the same
        #print newState[-1]
        newState1[-1] -= 1
        newState2[-1] -= 1

        newState1[-2] = index
        newState2[-2] = index
        return ((pNewState1, newState1), (pNewState2, newState2))
        #print newState
        #return newState

    
