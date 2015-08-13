from math import log
from random import sample
from numpy import dot, linalg
from numpy.random import randint
from utils import *
from copy import deepcopy
from ordereddict import OrderedDict


#this removes the example and then adds it back in.
def getChangeInClassifierRE(tasks, state, 
                                       classifier, accuracy, index):


    
    changeInClassifier0 = 0.0
    changeInClassifier1 = 0.0
    
    tempState0 = deepcopy(state)
    tempState1 = deepcopy(state)
    
    currentLabel = np.argmax(state[index])
    
    #REMOVE THE EXAMPLE AND THEN RETRAIN
    tempState0[index][0] = 0
    tempState0[index][1] = 0
    tempState1[index][0] = 0
    tempState1[index][1] = 0

    retrain(tasks, tempState0, classifier, True, accuracy)
    currentLabels = classifier.predict(tasks)

    tempState0[index][0] += 1
    tempState1[index][1] += 1
    
    predictedLabels0 = None
    predictedLabels1 = None

    if predictedLabels0 == None:
        retrain(tasks, tempState0, classifier, 
                True, accuracy)
        predictedLabels0 = classifier.predict(tasks)
    
    if predictedLabels1 == None:
        retrain(tasks, tempState1, classifier, 
                True, accuracy)
        predictedLabels1 = classifier.predict(tasks)

    for (predictedLabel0, predictedLabel1, label) in zip(predictedLabels0,
                                                         predictedLabels1,
                                                         currentLabels):
        if not predictedLabel0 == label:
            changeInClassifier0 += 1.0
        if not predictedLabel1 == label:
            changeInClassifier1 += 1.0
            
    nextLabelProbs = calcBayesProbability(
        state[index], accuracy, classifier.predict_proba(tasks[index])[0])
    #print "--"
    #print nextLabelProbs
    #print changeInClassifier0
    #print changeInClassifier1

    expectedChange = (
        (((nextLabelProbs[0] * accuracy) + 
         (nextLabelProbs[1] * (1.0-accuracy))) *
        changeInClassifier0) + 
        (((nextLabelProbs[1] * accuracy) +
         (nextLabelProbs[0] * (1.0-accuracy)))* 
        changeInClassifier1))

    return expectedChange

def getChangeInClassifierExpectedExpectedMax(
        tasks, state, classifier, accuracy, index):
    retrain(tasks, state, classifier, True, accuracy)
    currentLabels = classifier.predict(tasks)
    
    changeInClassifier0 = 0.0
    changeInClassifier1 = 0.0
    
    tempState0 = deepcopy(state)
    tempState1 = deepcopy(state)
    
    currentLabel = np.argmax(state[index])
    currentNotLabel = 1 - currentLabel

    if state[index][0] == state[index][1]:
        labelsNeededForChange = 1
    else:
        labelsNeededForChange = abs(state[index][0] - state[index][1])
    
    if currentLabel == 0:
        tempState0[index][0] += 1
        tempState1[index][1] += labelsNeededForChange
    elif currentLabel == 1:
        tempState0[index][0] += labelsNeededForChange
        tempState1[index][1] += 1
        
    
    predictedLabels0 = None
    predictedLabels1 = None

    if predictedLabels0 == None:
        retrain(tasks, tempState0, classifier, 
                True, accuracy)
        predictedLabels0 = classifier.predict(tasks)
    
    if predictedLabels1 == None:
        retrain(tasks, tempState1, classifier, 
                True, accuracy)
        predictedLabels1 = classifier.predict(tasks)

    for (predictedLabel0, predictedLabel1, label) in zip(predictedLabels0,
                                                         predictedLabels1,
                                                         currentLabels):
        if not predictedLabel0 == label:
            changeInClassifier0 += 1.0
        if not predictedLabel1 == label:
            changeInClassifier1 += 1.0
            
    nextLabelProbs = calcBayesProbability(
        state[index], accuracy, classifier.predict_proba(tasks[index])[0])

    #if not state[index][0] == 0 or not state[index][1] == 0:
    #print "-EEM-"
    #print state[index]
    #print nextLabelProbs
    #print changeInClassifier0
    #print changeInClassifier1
    #print labelsNeededForChange


    if currentLabel == 0:
        return (
            (((nextLabelProbs[0] * accuracy) + 
              (nextLabelProbs[1] * (1.0-accuracy))) * 
             changeInClassifier0) +
            (((nextLabelProbs[1] * accuracy) + 
             (nextLabelProbs[0] * (1.0-accuracy))) * 
             (changeInClassifier1 / labelsNeededForChange)))
    elif currentLabel == 1:
        return (
            (((nextLabelProbs[0] * accuracy) + 
              (nextLabelProbs[1] * (1.0-accuracy))) * 
             (changeInClassifier0 / labelsNeededForChange)) +
            (((nextLabelProbs[1] * accuracy) + 
              (nextLabelProbs[0] * (1.0-accuracy))) * 
             changeInClassifier1))


def getChangeInClassifierExpectedMax(tasks, state, classifier, accuracy, index):
    retrain(tasks, state, classifier, True, accuracy)
    currentLabels = classifier.predict(tasks)
    
    changeInClassifier0 = 0.0
    changeInClassifier1 = 0.0
    
    tempState0 = deepcopy(state)
    tempState1 = deepcopy(state)
    
    currentLabel = np.argmax(state[index])
    currentNotLabel = 1 - currentLabel

    labelsNeededForChange = abs(state[index][0] - state[index][1]) + 1
    
    tempState0[index][currentLabel] += 1
    tempState1[index][currentNotLabel] += labelsNeededForChange
    #tempState1[index][currentNotLabel] += 1

    
    predictedLabels0 = None
    predictedLabels1 = None

    if predictedLabels0 == None:
        retrain(tasks, tempState0, classifier, 
                True, accuracy)
        predictedLabels0 = classifier.predict(tasks)
    
    if predictedLabels1 == None:
        retrain(tasks, tempState1, classifier, 
                True, accuracy)
        predictedLabels1 = classifier.predict(tasks)

    for (predictedLabel0, predictedLabel1, label) in zip(predictedLabels0,
                                                         predictedLabels1,
                                                         currentLabels):
        if not predictedLabel0 == label:
            changeInClassifier0 += 1.0
        if not predictedLabel1 == label:
            changeInClassifier1 += 1.0
            
    nextLabelProbs = calcBayesProbability(
        state[index], accuracy, classifier.predict_proba(tasks[index])[0])

    #print "--EM--"
    #print state[index]
    #print nextLabelProbs
    #print changeInClassifier0
    #print changeInClassifier1

    maxChange = max(changeInClassifier0, changeInClassifier1)
    return maxChange / labelsNeededForChange

def getChangeInClassifierMax(tasks, state, classifier, accuracy, index):
    retrain(tasks, state, classifier, True, accuracy)
    currentLabels = classifier.predict(tasks)
    
    changeInClassifier0 = 0.0
    changeInClassifier1 = 0.0
    
    tempState0 = deepcopy(state)
    tempState1 = deepcopy(state)
    
    currentLabel = np.argmax(state[index])

    tempState0[index][0] += 1
    tempState1[index][1] += 1
    
    predictedLabels0 = None
    predictedLabels1 = None

    if predictedLabels0 == None:
        retrain(tasks, tempState0, classifier, 
                True, accuracy)
        predictedLabels0 = classifier.predict(tasks)
    
    if predictedLabels1 == None:
        retrain(tasks, tempState1, classifier, 
                True, accuracy)
        predictedLabels1 = classifier.predict(tasks)

    for (predictedLabel0, predictedLabel1, label) in zip(predictedLabels0,
                                                         predictedLabels1,
                                                         currentLabels):
        if not predictedLabel0 == label:
            changeInClassifier0 += 1.0
        if not predictedLabel1 == label:
            changeInClassifier1 += 1.0
            
    nextLabelProbs = calcBayesProbability(
        state[index], accuracy, classifier.predict_proba(tasks[index])[0])
    #print "--"
    #print nextLabelProbs
    #print changeInClassifier0
    #print changeInClassifier1

    maxChange = max(changeInClassifier0, changeInClassifier1)
    return maxChange


def doUnlabeledPseudoLookahead(tasks, task, state, classifier, baseStrategy,
                         dataGenerator, accuracy, k,
                         optimism = False):
    
    tempState0 = deepcopy(state)
    tempState1 = deepcopy(state)

    tempState0[task] = [0,0,k]
    tempState1[task] = [0,0,k]
    tempState0[task][0] += 1
    tempState1[task][1] += 1

    retrain(state, classifier, True, accuracy)
    nextLabelProbs = classifier.predict_proba(task)[0]
    currentLabels = classifier.predict(tasks)

    (trainingTasks0, trainingLabels0) = retrain(tempState0, 
                                                classifier, True, accuracy)

    predictedLabels0 = classifier.predict(tasks)

    (trainingTasks1, trainingLabels1) = retrain(tempState1, 
                                                classifier, True, accuracy)
    predictedLabels1 = classifier.predict(tasks)


    changeInClassifier0 = 0.0
    changeInClassifier1 = 0.0
    for (predictedLabel0, predictedLabel1, label) in zip(predictedLabels0,
                                                         predictedLabels1,
                                                         currentLabels):
        if not predictedLabel0 == label:
            changeInClassifier0 += 1.0
        if not predictedLabel1 == label:
            changeInClassifier1 += 1.0
            
    if optimism:
        expectedChange = max(changeInClassifier0, changeInClassifier1)
    else:
        expectedChange = (
            (((nextLabelProbs[0] * accuracy) + 
              (nextLabelProbs[1] * (1.0-accuracy))) *
             changeInClassifier0) + 
            (((nextLabelProbs[1] * accuracy) +
              (nextLabelProbs[0] * (1.0-accuracy)))* 
             changeInClassifier1))
        
    return expectedChange

#returns the impact after adding k points
def doUnlabeledLookahead(tasks, task, state, classifier, baseStrategy,
                         dataGenerator, currentLabels, accuracy, k,
                         optimism = False):
    
    tempState0 = deepcopy(state)
    tempState1 = deepcopy(state)

    tempState0[task] = [0,0]
    tempState1[task] = [0,0]
    tempState0[task][0] += 1
    tempState1[task][1] += 1

    retrain(state, classifier, True, accuracy)
    nextLabelProbs = classifier.predict_proba(task)[0]

    (trainingTasks0, trainingLabels0) = retrain(tempState0, 
                                                classifier, True, accuracy)

    nextTask0 = baseStrategy.sample(dataGenerator, tempState0, 
                                    classifier, accuracy)
    predictedLabels0 = classifier.predict(tasks)

    (trainingTasks1, trainingLabels1) = retrain(tempState1, 
                                                classifier, True, accuracy)

    nextTask1 = baseStrategy.sample(dataGenerator, tempState1, 
                                    classifier, accuracy)
    predictedLabels1 = classifier.predict(tasks)

    #print optimism

    if k == 0:
        changeInClassifier0 = 0.0
        changeInClassifier1 = 0.0
        for (predictedLabel0, predictedLabel1, label) in zip(predictedLabels0,
                                                             predictedLabels1,
                                                             currentLabels):
            if not predictedLabel0 == label:
                changeInClassifier0 += 1.0
            if not predictedLabel1 == label:
                changeInClassifier1 += 1.0

        if optimism:
            expectedChange = max(changeInClassifier0, changeInClassifier1)
        else:
            expectedChange = (
                (((nextLabelProbs[0] * accuracy) + 
                  (nextLabelProbs[1] * (1.0-accuracy))) *
                 changeInClassifier0) + 
                (((nextLabelProbs[1] * accuracy) +
                  (nextLabelProbs[0] * (1.0-accuracy)))* 
                 changeInClassifier1))
    else:
        if optimism:
            expectedChange = max(doUnlabeledLookahead(
                tasks, nextTask0, tempState0, 
                classifier, baseStrategy, dataGenerator, 
                currentLabels, accuracy, k-1, optimism),
             doUnlabeledLookahead(tasks, nextTask1, tempState1, 
                                  classifier, baseStrategy, dataGenerator, 
                                  currentLabels, accuracy, k-1, optimism))
        else:
            expectedChange = (
                (((nextLabelProbs[0] * accuracy) + 
                  (nextLabelProbs[1] * (1.0-accuracy))) *
                 doUnlabeledLookahead(tasks, nextTask0, tempState0, 
                                      classifier, baseStrategy, dataGenerator, 
                                      currentLabels, accuracy, k-1, optimism)) + 
                (((nextLabelProbs[1] * accuracy) +
                  (nextLabelProbs[0] * (1.0-accuracy)))* 
                 doUnlabeledLookahead(tasks, nextTask1, tempState1, 
                                      classifier, baseStrategy, dataGenerator, 
                                      currentLabels, accuracy, k-1, optimism)))

    return expectedChange

def getChangeInClassifier(tasks, state, classifier, accuracy, task,
                          optimism = False, pseudolookahead= False,
                          numBootstrapSamples = 0):
    
    changeInClassifier0 = 0.0
    changeInClassifier1 = 0.0
    
    tempState0 = deepcopy(state)
    tempState1 = deepcopy(state)

    if task not in state:
        tempState0[task] = [0,0]
        tempState1[task] = [0,0]
        currentLabel = 0
    else:
        currentLabel = np.argmax(state[task])

    #currentNotLabel = 1 - currentLabel

    if pseudolookahead and task in state:
        numAdditionalLabels = abs(state[task][0] - state[task][1]) + 1 
    else:
        #print "NOT THE WOODO"
        numAdditionalLabels = 1

    if currentLabel == 0:
        tempState0[task][0] += 1
        tempState1[task][1] += numAdditionalLabels
    elif currentLabel == 1:
        tempState0[task][0] += numAdditionalLabels
        tempState1[task][1] += 1

    predictedLabels0 = None
    predictedLabels1 = None

    #print "START RETRAINING"

    if predictedLabels0 == None:
        (trainingTasks0, trainingLabels0) = retrain(tempState0, classifier, True, accuracy)
        predictedLabels0 = classifier.predict(tasks)
    
    if predictedLabels1 == None:
        (trainingTasks1, trainingLabels1) = retrain(tempState1, classifier, True, accuracy)
        predictedLabels1 = classifier.predict(tasks)

    (trainingTasks, trainingLabels) = retrain(state, classifier, True, accuracy)
    currentLabels = classifier.predict(tasks)

    """
    if currentLabel == 0:
        tempState0[task][0] -= 1
        tempState1[task][1] -= numAdditionalLabels
    elif currentLabel == 1:
        tempState0[task][0] -= numAdditionalLabels
        tempState1[task][1] -= 1
    (trainingTasksTest, trainingLabelsTest) = retrain(tempState0, classifier, True, accuracy)
    predictedLabelsTest = classifier.predict(tasks)
    if currentLabel == 0:
        tempState0[task][0] += 1
        tempState1[task][1] += numAdditionalLabels
    elif currentLabel == 1:
        tempState0[task][0] += numAdditionalLabels
        tempState1[task][1] += 1
    """

    changeBetweenClassifiers = 0.0
    for (predictedLabel0, predictedLabel1, label) in zip(predictedLabels0,
                                                         predictedLabels1,
                                                         currentLabels):
        if not predictedLabel0 == label:
            changeInClassifier0 += 1.0
        if not predictedLabel1 == label:
            changeInClassifier1 += 1.0
        if not predictedLabel0 == predictedLabel1:
            changeBetweenClassifiers += 1.0

    changeInClassifier0 /= numAdditionalLabels
    changeInClassifier1 /= numAdditionalLabels

    if optimism:
        return max(changeInClassifier0, changeInClassifier1)


    if numBootstrapSamples != 0:
        priorSamples = []
        nonActiveTasks = state.keys()
        nonActiveTasks.remove(-1)
        baseStrategyChange = 0.0
        for numBootstrapSample in range(numBootstrapSamples): 
            bootstrapSampleTasks = sample(nonActiveTasks, 
                                          len(nonActiveTasks) / 2)
            bootstrapSampleState = OrderedDict(
                [(k, state[k]) for k in bootstrapSampleTasks])
            retrain(bootstrapSampleState, classifier, True, accuracy)
            priorSamples.append(classifier.predict_proba(task)[0])

    else:
        priorSamples = [classifier.predict_proba(task)[0]]

    expectedChange = 0.0    
    for priors in priorSamples:

        if task in state:
            nextLabelProbs = calcBayesProbability(
                state[task], accuracy, priors)
        else:
            nextLabelProbs = calcBayesProbability(
                [0,0], accuracy, priors)

        expectedChange += (
            (((nextLabelProbs[0] * accuracy) + 
              (nextLabelProbs[1] * (1.0-accuracy))) *
             changeInClassifier0) + 
            (((nextLabelProbs[1] * accuracy) +
              (nextLabelProbs[0] * (1.0-accuracy)))* 
             changeInClassifier1))

    expectedChange /= len(priors)

    """
    print "-Normal-"
    if task in state:
        print state[task]
    print tempState0[task]
    print tempState1[task]
    print len(state.keys())
    print len(tempState0.keys())
    print len(tempState1.keys())
    differences0 = 0.0
    differences1 = 0.0
   # print len(trainingDict.keys())
   # print len(trainingDict1.keys())
   # print len(trainingDict0.keys())
   # print len(trainingDictTest.keys())
    print sum(i != j for i,j in zip(predictedLabels0, predictedLabels1))
    print sum(i != j for i,j in zip(predictedLabels0, currentLabels))
    print sum(i != j for i,j in zip(predictedLabels1, currentLabels))
    #print sum(i != j for i,j in zip(predictedLabelsTest, currentLabels))
    #print sum(i != j for i,j in zip(predictedLabelsTest, predictedLabels0))
    #print sum(i != j for i,j in zip(predictedLabelsTest, predictedLabels1))

    
    #for t in trainingDict.keys():
    #    if tempState1[t][0] != tempState1[t][1]:
    print len(trainingTasks)
    print len(trainingTasks0)
    print len(trainingTasks1)
    #print len(trainingTasksTest)
    print len(trainingLabels)
    print len(trainingLabels0)
    print len(trainingLabels1)
    #print len(trainingLabelsTest)

    print trainingTasks == trainingTasks0
    #    if tempState0[t][0] != tempState0[t][1]:
    print trainingTasks == trainingTasks1
    #    if tempState0[t][0] != tempState0[t][1]:
   # print trainingTasks == trainingTasksTest
    print trainingTasks0 == trainingTasks1
    print trainingLabels == trainingLabels0
    #    if tempState0[t][0] != tempState0[t][1]:
    print trainingLabels == trainingLabels1
    #    if tempState0[t][0] != tempState0[t][1]:
    #print trainingLabels == trainingLabelsTest        
    print trainingLabels0 == trainingLabels1
    print nextLabelProbs

    print len(predictedLabels1)
    print len(currentLabels)
    print len(predictedLabels0)
    print changeInClassifier0
    print changeInClassifier1
    print changeBetweenClassifiers
    """

    return expectedChange


def getErrorInClassifier(tasks, state, classifier, accuracy, task,
                          optimism = False, pseudolookahead= False,
                          numBootstrapSamples = 0):
    
    
    tempState0 = deepcopy(state)
    tempState1 = deepcopy(state)

    if task not in state:
        tempState0[task] = [0,0]
        tempState1[task] = [0,0]
        currentLabel = 0
    else:
        currentLabel = np.argmax(state[task])

    #currentNotLabel = 1 - currentLabel

    if pseudolookahead and task in state:
        numAdditionalLabels = abs(state[task][0] - state[task][1]) + 1 
    else:
        #print "NOT THE WOODO"
        numAdditionalLabels = 1

    if currentLabel == 0:
        tempState0[task][0] += 1
        tempState1[task][1] += numAdditionalLabels
    elif currentLabel == 1:
        tempState0[task][0] += numAdditionalLabels
        tempState1[task][1] += 1


    #print "START RETRAINING"
    
    retrain(tempState0, classifier, True, accuracy)
    expectedError0 = calcExpectedError(tasks, classifier)
        
    retrain(tempState1, classifier, True, accuracy)
    expectedError1 = calcExpectedError(tasks, classifier)


    expectedError0 /= numAdditionalLabels
    expectedError1 /= numAdditionalLabels

    if optimism:
        return max(expectedError0, expectedError1)


    if numBootstrapSamples != 0:
        priorSamples = []
        nonActiveTasks = state.keys()
        nonActiveTasks.remove(-1)
        baseStrategyChange = 0.0
        for numBootstrapSample in range(numBootstrapSamples): 
            bootstrapSampleTasks = sample(nonActiveTasks, 
                                          len(nonActiveTasks) / 2)
            bootstrapSampleState = OrderedDict(
                [(k, state[k]) for k in bootstrapSampleTasks])
            retrain(bootstrapSampleState, classifier, True, accuracy)
            priorSamples.append(classifier.predict_proba(task)[0])

    else:
        priorSamples = [classifier.predict_proba(task)[0]]

    expectedError = 0.0    
    for priors in priorSamples:

        if task in state:
            nextLabelProbs = calcBayesProbability(
                state[task], accuracy, priors)
        else:
            nextLabelProbs = calcBayesProbability(
                [0,0], accuracy, priors)

        expectedError += (
            (((nextLabelProbs[0] * accuracy) + 
              (nextLabelProbs[1] * (1.0-accuracy))) *
             expectedError0) + 
            (((nextLabelProbs[1] * accuracy) +
              (nextLabelProbs[0] * (1.0-accuracy)))* 
             expectedError1))

    expectedError /= len(priors)

    return expectedError

def getAllUncertainties(examples, classifier):
    entropies = []
    probs = []
    probs = classifier.predict_proba(examples)
    
    #for taskIndex in taskIndices:
    #    probs.append(classifier.predict_proba(examples[taskIndex])[0])

        #print probs
    
    for prob in probs:
        #prob = probs[taskIndex]
        #for prob in probs:
        entropy = 0.0
        for p in prob:
            entropy += p * log(p+0.0000001)
            #print "BOOP"
            #print p
            #print log(p)
        #print entropy
        entropy *= -1
        entropies.append(entropy)

    return entropies

def getMostUncertainTask(tasks, classifier):
    highestUncertainty = -21930123123
    highestEntropyDistribution = None
    mostUncertainTaskIndices = []
    mustUncertainTasks = []

    entropies = getAllUncertainties(tasks, classifier)
    #for (task, i, uncertainty) in zip(tasks, taskIndices, entropies):    
    for (i, uncertainty) in zip(range(len(tasks)), entropies):    
    #for (i, uncertainty) in zip(range(len(entropies)), entropies):    
        task = tasks[i]
        if uncertainty > highestUncertainty:
            mostUncertainTaskIndices = [i]
            mostUncertainTasks = [task]
            highestUncertainty = uncertainty
        elif uncertainty == highestUncertainty:
            mostUncertainTaskIndices.append(i)
            mostUncertainTasks.append(task)

    #(mostUncertainTaskIndex, 
    # mostUncertainTask) = sample(zip(mostUncertainTaskIndices,
    #                               mostUncertainTasks), 1)[0]

    #j = sample(xrange(len(mostUncertainTasks)), 1)[0]
    j = randint(len(mostUncertainTasks))
    mostUncertainTaskIndex = mostUncertainTaskIndices[j]
    mostUncertainTask = mostUncertainTasks[j]

    return (classifier.predict_proba([mostUncertainTask])[0], 
            mostUncertainTaskIndex)


def getTotalUncertainty(examples, classifier):

    totalUncertainty = 0.0
    for example in examples:
        #print "YO"
        #print self.getUncertainty(example)
        totalUncertainty += classifier.getUncertainty(example)

    totalUncertainty /= len(examples)

    #return max(self.getAllUncertainties(examples))
    return totalUncertainty
