from copy import deepcopy
from utils import *
from classifierUtils import *
from math import exp, floor
from random import sample, randint
from ordereddict import OrderedDict

class samplingMethod:
    def __init__(self):
        pass

    def reinit(self):
        #Do nothing
        return 0

    def validate(self):
        return False

class uncertaintySampling(samplingMethod):

    def __init__(self):
        pass

    def reinit(self):
        pass

    def getName(self):
        return 'unc'

    def sample(self, dataGenerator, state, classifier, accuracy):
        retrain(state, classifier, True, accuracy)
        tasks = dataGenerator.trainingTasks
        (p, index) = getMostUncertainTask(tasks, classifier)
        return tasks[index]

class uncertaintySamplingLabeled(samplingMethod):

    def __init__(self):
        pass

    def reinit(self):
        pass

    def getName(self):
        return 'uncL'

    def sample(self, dataGenerator, state, classifier, accuracy):
        retrain(state, classifier, True, accuracy)
        nonActiveTasks = state.keys()
        nonActiveTasks.remove(-1)
        (p, index) = getMostUncertainTask(nonActiveTasks, classifier)
        return nonActiveTasks[index]


class impactSampling(samplingMethod):

    def __init__(self, optimism=False, pseudolookahead=False,
                 numBootstrapSamples = 0, symmetric = False,
                 getLastItemLabeled = False,
                 strategies = [uncertaintySampling(), 
                               uncertaintySamplingLabeled()]):
        self.optimism = optimism
        self.pseudolookahead = pseudolookahead
        self.symmetric = symmetric
        self.numBootstrapSamples = numBootstrapSamples
        self.baseStrategies = strategies
        self.outputString = ""
        self.lastLookaheadLength = 1
        self.currentLookaheadLength = 1

        self.getLastItemLabeled = getLastItemLabeled
        self.lastItemLabeled = None
        if self.getLastItemLabeled:
            self.baseStrategies.append('lastLabeled')

    def reinit(self):
        #self.optimism = self.optimism
        #self.pseudolookahead = self.pseudolookahead
        #self.baseStrategies = self.baseStrategies
        #self.__init__(self.optimism, self.pseudolookahead, self.baseStrategies)
        for baseStrategy in self.baseStrategies:
            if baseStrategy == 'lastLabeled':
                self.lastItemLabeled = None
                continue
            baseStrategy.reinit()
        self.logFile.write(self.outputString)
        self.logFile.write("\n")
        self.outputString = ""
        self.lastLookaheadLength = 1
        self.currentLookaheadLength = 1

    def getName(self):
        baseName = 'impactPrior'
        if self.numBootstrapSamples != 0:
            baseName += 'BOO'
        if self.pseudolookahead:
            baseName += 'PL'
        if self.optimism:
            baseName += 'OPT'
        if self.symmetric:
            baseName += '-S'
        baseName += '(%d)' % len(self.baseStrategies)

        return baseName

    def sample(self, dataGenerator, state, classifier, accuracy):

        tasks = dataGenerator.trainingTasks
        nonActiveTasks = state.keys()
        nonActiveTasks.remove(-1)
        allTasks = tasks + nonActiveTasks

        bestTasks = []
        bestChange = 0
        for (strategyNumber, baseStrategy) in zip(
                range(len(self.baseStrategies)), self.baseStrategies):
            if baseStrategy == 'lastLabeled':
                if self.lastItemLabeled == None:
                    continue
                baseStrategyTask = self.lastItemLabeled
            else:
                baseStrategyTask = baseStrategy.sample(
                    dataGenerator, state, classifier, accuracy)

            #If we are getting an OLD task, that is RELABELING
            if baseStrategyTask in nonActiveTasks:
                currentLookaheadLength = (max(state[baseStrategyTask]) -
                                            min(state[baseStrategyTask])) + 1
                if currentLookaheadLength > self.currentLookaheadLength:
                    self.currentLookaheadLength = currentLookaheadLength
                baseStrategyChange = max(
                    getChangeInClassifier(
                        allTasks, state, classifier, accuracy, baseStrategyTask,
                        optimism = self.optimism, 
                        pseudolookahead = self.pseudolookahead,
                        flipLabel = True,
                        numBootstrapSamples = self.numBootstrapSamples),
                    getChangeInClassifier(
                        allTasks, state, classifier, accuracy, baseStrategyTask,
                        optimism = self.optimism, 
                        pseudolookahead = self.pseudolookahead,
                        flipLabel = False,
                        numBootstrapSamples = self.numBootstrapSamples))

            else:
                if self.symmetric:
                    """
                    if baseStrategyChange == 0  and self.pseudolookahead:
                    """ 
                    bestPseudolookahead = 0.0
                    if self.pseudolookahead:
                        print "DOING UNLABELED LOOKAHEAD"
                        for length in range(1,self.lastLookaheadLength+1):
                            baseStrategyChange = doUnlabeledPseudoLookahead(
                                tasks, baseStrategyTask, state, classifier, 
                                baseStrategy, dataGenerator,
                                accuracy, length,
                                optimism = self.optimism)
                            baseStrategyChange /= length
                            #print baseStrategyChange
                            #print bestPseudolookahead
                            #print baseStrategyChange > bestPseudolookahead
                            if baseStrategyChange > bestPseudolookahead:
                                bestPseudolookahead = baseStrategyChange
                        baseStrategyChange = bestPseudolookahead
                    else:
                        baseStrategyChange = getChangeInClassifier(
                            allTasks, state, classifier, 
                            accuracy, baseStrategyTask,
                            optimism = self.optimism, 
                            pseudolookahead = self.pseudolookahead,
                            numBootstrapSamples = self.numBootstrapSamples)


                else:
                    baseStrategyChange = getChangeInClassifier(
                        allTasks, state, classifier, accuracy, baseStrategyTask,
                        optimism = self.optimism,
                        numBootstrapSamples = self.numBootstrapSamples)


            print baseStrategy
            print baseStrategyChange
            if baseStrategyTask in state:
                print state[baseStrategyTask]
            
            if baseStrategyChange > bestChange:
                bestTasks = [(strategyNumber, baseStrategyTask)]
                bestChange = baseStrategyChange
            elif baseStrategyChange == bestChange:
                bestTasks.append((strategyNumber, baseStrategyTask))
            else:
                continue
                
        self.lastLookaheadLength = self.currentLookaheadLength
        self.currentLookaheadLength = 1
        print "Best tasks"
        print [x for (x,y)  in bestTasks]
        (nextStrategyNumber, nextTask) = sample(bestTasks,1)[0]
        print nextStrategyNumber
        self.outputString += ("%f\t" % nextStrategyNumber)
        self.lastItemLabeled = nextTask
        return nextTask

        """

        (p, ALIndex) = getMostUncertainTask(tasks, classifier)
        (p, RLIndex) = getMostUncertainTask(nonActiveTasks, classifier)

        #changeFromRelabeling = getChangeInClassifier(
        #    tasks, state, classifier, accuracy, self.lastLabelIndex)
        changeFromAL = getChangeInClassifier(
            allTasks, state, classifier, accuracy, tasks[ALIndex])

        changeFromRelabeling = getChangeInClassifier(
            allTasks, state, classifier, accuracy, nonActiveTasks[RLIndex],
            optimism = self.optimism, pseudolookahead = self.pseudolookahead)
        """

        #print "IMPACT SAMPLING"
        #print RLIndex
        #print changeFromRelabeling
        #print state[nonActiveTasks[RLIndex]]
        #print changeFromAL

        """
        if changeFromRelabeling > changeFromAL:
            #return self.lastLabelIndex
            #print "R"
            return nonActiveTasks[RLIndex]
        else:
            #print "A"
            self.lastLabelIndex = ALIndex
            return tasks[ALIndex]
        """

class impactSamplingAll(samplingMethod):

    def __init__(self, optimism=False, pseudolookahead=False,
                 numBootstrapSamples = 0, symmetric = False):
        self.optimism = optimism
        self.pseudolookahead = pseudolookahead
        self.symmetric = symmetric
        self.numBootstrapSamples = numBootstrapSamples

    def reinit(self):
        pass

    def getName(self):
        baseName = 'impactPrior'
        if self.numBootstrapSamples != 0:
            baseName += 'BOO'
        if self.pseudolookahead:
            baseName += 'PL'
        if self.optimism:
            baseName += 'OPT'
        if self.symmetric:
            baseName += '-S'
        baseName += '(*)'

        return baseName

    def sample(self, dataGenerator, state, classifier, accuracy):

        tasks = dataGenerator.trainingTasks
        nonActiveTasks = state.keys()
        nonActiveTasks.remove(-1)
        allTasks = tasks + nonActiveTasks

        bestTasks = []
        bestChange = 0
        for task in allTasks:
            if task in tasks:
                if self.symmetric:
                    baseStrategyChange = getChangeInClassifier(
                        allTasks, state, classifier, accuracy, task,
                        optimism = self.optimism, 
                        pseudolookahead = self.pseudolookahead,
                        numBootstrapSamples = self.numBootstrapSamples)
                else:
                    baseStrategyChange = getChangeInClassifier(
                        allTasks, state, classifier, accuracy, task,
                        numBootstrapSamples = self.numBootstrapSamples)
            else:
                baseStrategyChange = getChangeInClassifier(
                    allTasks, state, classifier, accuracy, task,
                    optimism = self.optimism, 
                    pseudolookahead = self.pseudolookahead,
                    numBootstrapSamples = self.numBootstrapSamples)


            #print baseStrategyChange
            #if task in state:
            #    print state[task]
            
            if baseStrategyChange > bestChange:
                bestTasks = [task]
                bestChange = baseStrategyChange
            elif baseStrategyChange == bestChange:
                bestTasks.append(task)
            else:
                continue

        nextTask = sample(bestTasks,1)[0]
        return nextTask

class passive(samplingMethod):

    def __init__(self):
        pass

    def reinit(self):
        pass

    def getName(self):
        return 'pass'

    def sample(self, dataGenerator, state, classifier, accuracy):
        #activeTaskIndices.remove(len(activeTaskIndices)- 1)
        #return activeTaskIndices[len(activeTaskIndices)-1]
        #return activeTaskIndices.pop(-1)
        return sample(dataGenerator.trainingTasks, 1)[0]


class passive(samplingMethod):

    def __init__(self):
        pass

    def reinit(self):
        pass

    def getName(self):
        return 'pass'

    def sample(self, dataGenerator, state, classifier, accuracy):
        #activeTaskIndices.remove(len(activeTaskIndices)- 1)
        #return activeTaskIndices[len(activeTaskIndices)-1]
        #return activeTaskIndices.pop(-1)
        return sample(dataGenerator.trainingTasks, 1)[0]


class passiveAll(samplingMethod):

    def __init__(self):
        pass

    def reinit(self):
        pass

    def getName(self):
        return 'passAll'

    def sample(self, dataGenerator, state, classifier, accuracy):
        tasks = dataGenerator.trainingTasks
        nonActiveTasks = state.keys()
        nonActiveTasks.remove(-1)
        allTasks = tasks + nonActiveTasks

        return sample(allTasks, 1)[0]


#This is not passive. This is randomly choosing between two points.
class randomSampling(samplingMethod):

    def __init__(self, 
                 strategies = [uncertaintySampling(), 
                               uncertaintySamplingLabeled()]):
        self.baseStrategies = strategies

    def reinit(self):
        for baseStrategy in self.baseStrategies:
            baseStrategy.reinit()

    def getName(self):
        return 'random(%d)' % len(self.baseStrategies)

    def sample(self, dataGenerator, state, classifier, accuracy):

        bestTasks = []
        for (strategyNumber, baseStrategy) in zip(
                range(len(self.baseStrategies)), self.baseStrategies):
            baseStrategyTask = baseStrategy.sample(
                dataGenerator, state, classifier, accuracy)

            bestTasks.append((strategyNumber, baseStrategyTask))

        (nextStrategyNumber, nextTask) = sample(bestTasks,1)[0]
        return nextTask

class uncertaintySamplingRelabel(samplingMethod):

    def __init__(self, r):
        self.lastTask = None
        self.numRelabels = r

    def reinit(self):
        self.lastTask = None

    def getName(self):
        return 'unc-r%d' % self.numRelabels

    def sample(self, dataGenerator, state, classifier, accuracy):
        #index = classifier.getMostUncertainTask(tasks, activeTaskIndices)

        if not self.lastTask == None:
            if (state[self.lastTask][0] <= int(self.numRelabels / 2) and
                state[self.lastTask][1] <= int(self.numRelabels / 2)):
                return self.lastTask

        retrain(state, classifier, True, accuracy)
        tasks = dataGenerator.trainingTasks
        (p, index) = getMostUncertainTask(tasks, classifier)
        self.lastTask = tasks[index]
        return self.lastTask

class uncertaintySamplingAlpha(samplingMethod):
    
    def __init__(self,a):
        self.alpha = a

    def reinit(self):
        pass
        
    def getName(self):
        return 'unc%.1f' % self.alpha

    def sample(self, dataGenerator, state, classifier, accuracy):

        retrain(state, classifier, True, accuracy)
        unlabeledTasks = dataGenerator.trainingTasks
        labeledTasks = state.keys()
        labeledTasks.remove(-1)
        allTasks = unlabeledTasks + labeledTasks

        highestScore = -21930123123
        mustUncertainTasks = []

        entropies = getAllUncertainties(allTasks, classifier)

        for (classifierEntropy, task) in zip(entropies, allTasks):

            if task in state:
                (pZero, pOne) = calcBayesProbability(state[task], accuracy)
            else:
                (pZero, pOne) = calcBayesProbability([0,0], accuracy)

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

            totalEntropy = (((1.0-self.alpha) * classifierEntropy) +
                            (self.alpha * labelEntropy))

            #print "ENTROPIES"
            #print state[activeTaskIndex]
            #print labelEntropy
            #print classifierEntropy
            #print totalEntropy

            if totalEntropy > highestScore:
                mostUncertainTasks = [task]
                highestScore = totalEntropy
            elif totalEntropy == highestScore:
                mostUncertainTasks.append(task)

        #print mostUncertainTaskIndices
        nextTask = sample(mostUncertainTasks, 1)[0]

        return nextTask

class bayesianUncertaintySampling(samplingMethod):
    
    def __init__(self):
        pass

    def reinit(self):
        pass
        
    def getName(self):
        return 'uncBayes'

    def sample(self, activeTaskIndices, tasks, 
                             validationTasks, state, classifier,
                             accuracy):
        highestScore = -21930123123
        highestEntropyDistribution = None
        mostUncertainTaskIndices = []
        mustUncertainTasks = []

        for activeTaskIndex in activeTaskIndices:
            task = tasks[activeTaskIndex]

            (pZero, pOne) = calcBayesProbability(
                state[activeTaskIndex], 
                accuracy,
                classifier.predict_proba(task)[0])

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


            #print "ENTROPIES"
            #print state[activeTaskIndex]
            #print labelEntropy

            if labelEntropy > highestScore:
                mostUncertainTaskIndices = [activeTaskIndex]
                mostUncertainTasks = [task]
                highestScore = labelEntropy
            elif labelEntropy == highestScore:
                mostUncertainTaskIndices.append(activeTaskIndex)
                mostUncertainTasks.append(task)

        nextTaskIndex = sample(mostUncertainTaskIndices, 1)[0]

        return nextTaskIndex
class uncertaintySamplingAlphaRelabel(samplingMethod):
    
    def __init__(self,a, r):
        self.alpha = a
        self.numRelabels = r

    def reinit(self):
        pass
        
    def getName(self):
        return 'unc%.1f-r%d' % (self.alpha, self.numRelabels)

    def sample(self, activeTaskIndices, tasks, 
                             validationTasks, state, classifier,
                             accuracy):
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

            totalEntropy = (((1.0-self.alpha) * classifierEntropy) +
                            (self.alpha * labelEntropy))

            #print "ENTROPIES"
            #print state[activeTaskIndex]
            #print labelEntropy
            #print classifierEntropy
            #print totalEntropy

            if totalEntropy > highestScore:
                mostUncertainTaskIndices = [activeTaskIndex]
                mostUncertainTasks = [task]
                highestScore = totalEntropy
            elif totalEntropy == highestScore:
                mostUncertainTaskIndices.append(activeTaskIndex)
                mostUncertainTasks.append(task)

        nextTaskIndex = sample(mostUncertainTaskIndices, 1)[0]
        """
        task = tasks[nextTaskIndex]
        (pZero, pOne) = calcBayesProbability(state[nextTaskIndex], 
                                             accuracy)

        if pZero < 0.0001:
            pZero = 0.0001
            pOne = 0.9999
        if pOne < 0.0001:
            pOne = 0.0001
            pZero = 0.9999

        labelEntropy = (pZero * log(pZero)) + (pOne * log(pOne))
        labelEntropy *= -1.0
        
        totalEntropy = (((1.0-self.alpha) * classifierEntropy) +
                        (self.alpha * labelEntropy))

        print "ENTROPIES"
        print state[nextTaskIndex]
        print labelEntropy
        print pZero
        print pOne
        print classifierEntropy
        print totalEntropy
        """

        return nextTaskIndex



class DTVOISampling(samplingMethod):

    def __init__(self, optimism=False, pseudolookahead=False,
                 numBootstrapSamples = 0, symmetric = False,
                 getLastItemLabeled = False,
                 strategies = [uncertaintySampling(), 
                               uncertaintySamplingLabeled()]):
        self.optimism = optimism
        self.pseudolookahead = pseudolookahead
        self.symmetric = symmetric
        self.numBootstrapSamples = numBootstrapSamples
        self.baseStrategies = strategies
        self.outputString = ""
        self.lastLookaheadLength = 1
        self.currentLookaheadLength = 1

        self.getLastItemLabeled = getLastItemLabeled
        self.lastItemLabeled = None
        if self.getLastItemLabeled:
            self.baseStrategies.append('lastLabeled')

    def reinit(self):
        #self.optimism = self.optimism
        #self.pseudolookahead = self.pseudolookahead
        #self.baseStrategies = self.baseStrategies
        #self.__init__(self.optimism, self.pseudolookahead, self.baseStrategies)
        for baseStrategy in self.baseStrategies:
            if baseStrategy == 'lastLabeled':
                self.lastItemLabeled = None
                continue
            baseStrategy.reinit()
        self.logFile.write(self.outputString)
        self.logFile.write("\n")
        self.outputString = ""
        self.lastLookaheadLength = 1
        self.currentLookaheadLength = 1

    def getName(self):
        baseName = 'dtvoi'
        if self.numBootstrapSamples != 0:
            baseName += 'BOO'
        if self.pseudolookahead:
            baseName += 'PL'
        if self.optimism:
            baseName += 'OPT'
        if self.symmetric:
            baseName += '-S'
        baseName += '(%d)' % len(self.baseStrategies)

        return baseName

    def sample(self, dataGenerator, state, classifier, accuracy):

        tasks = dataGenerator.trainingTasks
        nonActiveTasks = state.keys()
        nonActiveTasks.remove(-1)
        allTasks = tasks + nonActiveTasks

        bestTasks = []
        bestError = float('Inf')
        for (strategyNumber, baseStrategy) in zip(
                range(len(self.baseStrategies)), self.baseStrategies):
            if baseStrategy == 'lastLabeled':
                if self.lastItemLabeled == None:
                    continue
                baseStrategyTask = self.lastItemLabeled
            else:
                baseStrategyTask = baseStrategy.sample(
                    dataGenerator, state, classifier, accuracy)

            #If we are getting an OLD task
            if baseStrategyTask in nonActiveTasks:
                currentLookaheadLength = (max(state[baseStrategyTask]) -
                                            min(state[baseStrategyTask])) + 1
                if currentLookaheadLength > self.currentLookaheadLength:
                    self.currentLookaheadLength = currentLookaheadLength
                baseStrategyChange = getErrorInClassifier(
                    allTasks, state, classifier, accuracy, baseStrategyTask,
                    optimism = self.optimism, 
                    pseudolookahead = self.pseudolookahead,
                    numBootstrapSamples = self.numBootstrapSamples)

            else:
                if self.symmetric:
                    baseStrategyChange = getErrorInClassifier(
                        allTasks, state, classifier, 
                        accuracy, baseStrategyTask,
                        optimism = self.optimism, 
                        pseudolookahead = self.pseudolookahead,
                        numBootstrapSamples = self.numBootstrapSamples)
                    if baseStrategyChange == 0  and self.pseudolookahead:
                        print "DOING UNLABELED LOOKAHEAD"
                        baseStrategyChange = doUnlabeledPseudoLookahead(
                            tasks, baseStrategyTask, state, classifier, 
                            baseStrategy, dataGenerator,
                            accuracy, self.lastLookaheadLength,
                            optimism = self.optimism)
                        baseStrategyChange /= self.lastLookaheadLength

                else:
                    baseStrategyChange = getErrorInClassifier(
                        allTasks, state, classifier, accuracy, baseStrategyTask,
                        numBootstrapSamples = self.numBootstrapSamples)


            print baseStrategy
            print baseStrategyChange
            if baseStrategyTask in state:
                print state[baseStrategyTask]
            
            if baseStrategyChange < bestError:
                bestTasks = [(strategyNumber, baseStrategyTask)]
                bestError = baseStrategyChange
            elif baseStrategyChange == bestError:
                bestTasks.append((strategyNumber, baseStrategyTask))
            else:
                continue
                
        self.lastLookaheadLength = self.currentLookaheadLength
        self.currentLookaheadLength = 1
        print "Best tasks"
        print [x for (x,y)  in bestTasks]
        (nextStrategyNumber, nextTask) = sample(bestTasks,1)[0]
        print nextStrategyNumber
        self.outputString += ("%f\t" % nextStrategyNumber)
        self.lastItemLabeled = nextTask
        return nextTask

        """

        (p, ALIndex) = getMostUncertainTask(tasks, classifier)
        (p, RLIndex) = getMostUncertainTask(nonActiveTasks, classifier)

        #changeFromRelabeling = getChangeInClassifier(
        #    tasks, state, classifier, accuracy, self.lastLabelIndex)
        changeFromAL = getChangeInClassifier(
            allTasks, state, classifier, accuracy, tasks[ALIndex])

        changeFromRelabeling = getChangeInClassifier(
            allTasks, state, classifier, accuracy, nonActiveTasks[RLIndex],
            optimism = self.optimism, pseudolookahead = self.pseudolookahead)
        """

        #print "IMPACT SAMPLING"
        #print RLIndex
        #print changeFromRelabeling
        #print state[nonActiveTasks[RLIndex]]
        #print changeFromAL

        """
        if changeFromRelabeling > changeFromAL:
            #return self.lastLabelIndex
            #print "R"
            return nonActiveTasks[RLIndex]
        else:
            #print "A"
            self.lastLabelIndex = ALIndex
            return tasks[ALIndex]
        """
