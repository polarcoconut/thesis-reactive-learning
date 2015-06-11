from copy import deepcopy
from utils import *
from classifierUtils import *
from math import exp

def passive(dataGenerator,state, classifier, accuracy):
    
    #activeTaskIndices.remove(len(activeTaskIndices)- 1)
    #return activeTaskIndices[len(activeTaskIndices)-1]
    return sample(dataGenerator.trainingTasks, 1)[0]


def uncertaintySampling(activeTaskIndices, tasks, 
                        validationTasks, state, classifier,
                        accuracy):
    #index = classifier.getMostUncertainTask(tasks, activeTaskIndices)

    (p, index) = getMostUncertainTask(tasks, activeTaskIndices, classifier)
    activeTaskIndices.remove(index) #Only if we don't want to relabel
    return index


def uncertaintySamplingAlpha(a):
#This version adds the possibility of relabeling
    def uncertaintySampling2(activeTaskIndices, tasks, 
                             validationTasks, state, classifier,
                             accuracy, alpha = a):
        highestScore = -21930123123
        highestEntropyDistribution = None
        mostUncertainTaskIndices = []
        mustUncertainTasks = []

        #index = classifier.getMostUncertainTask(tasks, activeTaskIndices)

        entropies = getAllUncertainties(tasks, activeTaskIndices, classifier)

        for (classifierEntropy, activeTaskIndex) in zip(entropies, 
                                                        activeTaskIndices):
            task = tasks[activeTaskIndex]
            (pZero, pOne) = calcBayesProbability(state[activeTaskIndex], 
                                                 accuracy)

            #print state[activeTaskIndex]
            #print pZero
            #print pOne
            if pZero < 0.0001:
                pZero = 0.0001
                pOne = 0.9999
            if pOne < 0.0001:
                pOne = 0.0001
                pZero = 0.9999

            labelEntropy = (pZero * log(pZero)) + (pOne * log(pOne))
            labelEntropy *= -1.0

            totalEntropy = (((1.0-alpha) * classifierEntropy) +
                            (alpha * labelEntropy))

            print "ENTROPIES"
            #print state[activeTaskIndex]
            print labelEntropy
            print classifierEntropy

            if totalEntropy > highestScore:
                mostUncertainTaskIndices = [activeTaskIndex]
                mostUncertainTasks = [task]
                highestScore = totalEntropy
            elif totalEntropy == highestScore:
                mostUncertainTaskIndices.append(activeTaskIndex)
                mostUncertainTasks.append(task)


        return mostUncertainTaskIndices[0]
    return uncertaintySampling2
 

def minimizeExpectedError(activeTaskIndices, tasks, classifier):
    pass


def myopicVOI(activeTaskIndices, trainingTasks,
              validationTasks, 
              state, classifier,
              accuracy):
    bestRisk = float('inf')
    bestTaskIndex = None
    bestTask = None

    print classifier

    #We are going to assume that the classifier does not need retraining here
    #retrain(trainingTasks, state[0:-1], classifier,
    #        True, accuracy)
    classes = classifier.classifier.classes_
    numClasses = len(classes)


    for activeTaskIndex in activeTaskIndices:
        #print activeTaskIndex
        activeTask = trainingTasks[activeTaskIndex]
        probs = classifier.predict_proba(activeTask)[0]

        totalRisk = 0.0
        #print totalRisk
        for (prob, label) in zip(probs, classes):

            #this is the probability it will get labeled as label
            pLabel = prob * accuracy + (1.0 - prob) * (1.0 - accuracy)
            stateCopy = deepcopy(state)
            #print stateCopy
            classifierCopy = deepcopy(classifier)
            #print "HERE"
            stateCopy[activeTaskIndex][label] += 1
            #print "THERE"
            #retrain(trainingTasks, stateCopy[0:-1], classifier,
            #        True, accuracy)
            update(activeTask, stateCopy[activeTaskIndex],
                   classifierCopy, accuracy)

            
            risk = 0.0
            #print risk
            for validationTask in validationTasks:
                probsValidationTask = classifierCopy.predict_proba(
                    validationTask)[0]
                risk += (2.0 * (1.0 - probsValidationTask[0]) * 
                         probsValidationTask[0])
                #print probsValidationTask[0]
                #print risk

            totalRisk += (pLabel * risk)

        #print totalRisk
        if totalRisk < bestRisk:
            bestRisk = totalRisk
            bestTaskIndex = activeTaskIndex 
            

    return bestTaskIndex

#def simulatedAnnealing(temperature = 10000, coolingRate = 0.003):
def simulatedAnnealing():
    #temperature2 = 10000
    temp = 10000
    coolingRate = 0.003
    print temp
    def simulatedAnnealingHelper(activeTaskIndices, trainingTasks,
                           validationTasks, 
                           state, classifier,
                           accuracy):
        print coolingRate
        print temp
        bestRisk = float('inf')
        bestTaskIndex = None
        bestTask = None

        print classifier

        #We are going to assume that the classifier does not need retraining here
        #retrain(trainingTasks, state[0:-1], classifier,
        #        True, accuracy)
        classes = classifier.classifier.classes_
        numClasses = len(classes)

        while True:
            neighborIndex = sample(activeTaskIndices, 1)[0]
            neighbor = trainingTasks[neighborIndex]
            probs = classifier.predict_proba([neighbor])[0]

            oldRisk = 0.0
            #print risk
            for validationTask in validationTasks:
                probsValidationTaskOld = classifier.predict_proba(
                    [validationTask])[0]
                oldRisk += (2.0 * (1.0 - probsValidationTaskOld[0]) * 
                            probsValidationTaskOld[0])

            totalRisk = 0.0
            #print totalRisk
            for (prob, label) in zip(probs, classes):
                #this is the probability it will get labeled as label
                pLabel = prob * accuracy + (1.0 - prob) * (1.0 - accuracy)
                stateCopy = deepcopy(state)
                #print stateCopy
                classifierCopy = deepcopy(classifier)
                #print "HERE"
                stateCopy[neighborIndex][label] += 1
                #print "THERE"
                retrain(trainingTasks, stateCopy[0:-1], classifier,
                        True, accuracy)


                risk = 0.0
                #print risk
                for validationTask in validationTasks:
                    probsValidationTask = classifierCopy.predict_proba(
                        validationTask)[0]
                    risk += (2.0 * (1.0 - probsValidationTask[0]) * 
                         probsValidationTask[0])
                    #print probsValidationTask[0]
                    #print risk

                totalRisk += (pLabel * risk)

            #print totalRisk
            if totalRisk < oldRisk:
                temp *= (1.0 - coolingRate)
                return neighborIndex
            else:
                if random() < exp((oldRisk - totalRisk) / temp):
                    temp *= (1.0 - coolingRate)
                    return neighborIndex
                temp *= (1.0 - coolingRate)

        return bestTaskIndex
    return simulatedAnnealingHelper
