from random import random
from copy import deepcopy
from us import *
from utils import *

class MDP:


    #each task is a feature vector
    #we consider true = 1 and false = 0
    def __init__(self, trainingTasks, testingTasks, classifier, budget,
                 priorTrue):
        
        self.priorTrue = priorTrue
        self.priorFalse = 1.0 - priorTrue
        self.tasks = trainingTasks
        self.testingTasks = testingTasks
        self.numberOfTasks = len(trainingTasks)
        self.classifier = classifier
        self.budget = budget


    #the state is a list of trues and falses and then the current budget
    def getInitialState(self):
        initialState = [(0,0) for i in range(self.numberOfTasks)] 
        initialState.append(self.budget)
        return initialState

    #-1 means get a label for a new task
    # 0 ... n means get a label for an existing task
    def getActions(self, state):
        actions = [-1]
        for (i, (x, y)) in zip(range(self.numberOfTasks), state[0:-1]):
            if x > 0 or y > 0:
                actions.append(i)

        return actions

    def isTerminal(self, state):
        if state[-1] <= 0:
            return True
        else:
            return False


    #In this MDP, rewards are awarded upon taking an action and 
    #arriving at a state. Rewards are
    # only awarded when the budget is exhausted
    def getReward(self, state):
        currentBudget = state[-1]
        if currentBudget > 0:
            return 0
        else:
            return self.evalClassifier(state)

    def getHeuristic(self, state, action):
        newTrueState = deepcopy(state)
        newFalseState = deepcopy(state)


        if action == -1:
            activeTasks = []
            activeTaskIndices = []
            for (i, task, (trues, falses)) in zip(range(len(self.tasks)), 
                                                  self.tasks, state[0:-1]):
                if trues == 0 and falses == 0:
                    activeTasks.append(task)
                    activeTaskIndices.append(i)
                    
            self.classifier = retrain(self.tasks, state[0:-1], self.classifier)
                    
            #print getMostUncertainTask(
            #    self.classifier, 
            #    activeTasks,
            #    activeTaskIndices) 
            ((pFalse, pTrue), mostUncertainTaskIndex) = getMostUncertainTask(
                self.classifier, 
                activeTasks,
                activeTaskIndices) 

            newTrueState[mostUncertainTaskIndex] = (1, 0)
            newFalseState[mostUncertainTaskIndex] = (0, 1)

        else:
            (trues, falses) = state[action]
            pTrue = (trues + 1.0) / (trues + falses + 2.0)
            pFalse = (falses + 1.0) / (trues + falses + 2.0)

            newTrueState[action] = (trues+1, falses)
            newFalseState[action] = (trues, falses+1)
            
        #print self.evalClassifier(newTrueState)
        #print pTrue
        #print pFalse
        print ((pTrue * self.evalClassifier(newTrueState)) + 
                (pFalse * self.evalClassifier(newFalseState)))
        return ((pTrue * self.evalClassifier(newTrueState)) + 
                (pFalse * self.evalClassifier(newFalseState)))

    def evalClassifier(self, state):

        self.classifier = retrain(self.tasks, state[0:-1], self.classifier)

        numberFalsePredictions = 0.0
        for testingTask in self.testingTasks:
            prediction = self.classifier.predict([testingTask])
            if prediction == 0:
                numberFalsePredictions += 1.0
        #print numberFalsePredictions
        #print len(self.testingTasks)
        #print self.priorTrue
        #print (2.0 * self.priorTrue - 1.0)
        #print (numberFalsePredictions / len(self.testingTasks))
        #print (self.priorTrue - (float(numberFalsePredictions) / len(self.testingTasks)))
        risk = 1.0 - ((self.priorTrue - (numberFalsePredictions / 
                                        len(self.testingTasks))) / 
                      (2.0 * self.priorTrue - 1.0))
        print "RISK"
        print risk

        return -1.0 * risk

                    
    #this transition function does not mutate the state
    def T(self, state, action):
        
        newState = deepcopy(state)
        
        #If the action is to get a label for a new task
        if action == -1:
            activeTasks = []
            activeTaskIndices = []
            for (i, task, (trues, falses)) in zip(range(len(self.tasks)), 
                                                  self.tasks, state[0:-1]):
                if trues == 0 and falses == 0:
                    activeTasks.append(task)
                    activeTaskIndices.append(i)
                    

            self.classifier = retrain(self.tasks, state[0:-1], self.classifier)
        
            ((pFalse, pTrue), mostUncertainTaskIndex) = getMostUncertainTask(
                self.classifier, 
                activeTasks,
                activeTaskIndices)

            mostUncertainTask = self.tasks[mostUncertainTaskIndex]
            r = random()
            if r < pTrue:
                newState[mostUncertainTaskIndex] = (1, 0)
            else:
                newState[mostUncertainTaskIndex] = (0, 1)
                
        else:
            (trues, falses) = state[action]
            pTrue = (trues + 1.0) / (trues  + falses + 2.0)
            r = random()
            if r < pTrue:
                newState[action] = (trues+1, falses)
            else:
                newState[action] = (trues, falses+1)
            
        #for now, each task costs the same
        #print newState[-1]
        newState[-1] -= 1
        print newState
        return newState

    
