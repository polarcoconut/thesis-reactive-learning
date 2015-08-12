from random import randint, random, choice, seed, sample, shuffle
from numpy.linalg import norm
import numpy as np
from math import ceil, floor
from copy import deepcopy
from angli1 import getFeatures, getTestAndGoldData, getTrainingData_2, trainAndTestModel_2
from angli2 import *

class dataGenerator:
    def __init__(self, numExamples, numFeatures, pool = True):
        self.numExamples = numExamples
        self.numTrainingTasks = int(floor(0.7 * self.numExamples))
        self.numValidationTasks = int(floor(0.15 * self.numExamples))
        self.numTestingTasks = int(floor(0.15 *self.numExamples))
        self.numFeatures = numFeatures
        self.pool = pool

    def reinit(self):
        self.__init__()
        
    def splitData(self, instances, classes):
        self.oldTrainingTasks = instances[0:self.numTrainingTasks]
        self.oldTrainingTaskClasses = {}
        for (trainingTask, trainingTaskClass) in zip(
                self.oldTrainingTasks, classes[0:self.numTrainingTasks]):
            self.oldTrainingTaskClasses[trainingTask] = trainingTaskClass
        
        #self.trainingTaskClasses = classes[0:self.numTrainingTasks]
        self.oldTrainingTaskDifficulties = [
            0.5 for i in range(self.numTrainingTasks)]

        self.oldValidationTasks = instances[
            self.numTrainingTasks:self.numTrainingTasks+self.numValidationTasks]
        self.oldValidationTaskClasses = classes[
            self.numTrainingTasks:self.numTrainingTasks+self.numValidationTasks]
        
        self.oldTestingTasks = instances[
            self.numTrainingTasks+self.numValidationTasks:]
        self.oldTestingTaskClasses = classes[
            self.numTrainingTasks+self.numValidationTasks:]

        self.generateDuplicateData()

    def generateDuplicateData(self):
        self.trainingTasks = deepcopy(self.oldTrainingTasks)
        self.trainingTaskClasses = deepcopy(self.oldTrainingTaskClasses)
        self.trainingTaskDifficulties = deepcopy(
            self.oldTrainingTaskDifficulties)
        self.validationTasks = deepcopy(self.oldValidationTasks)
        self.validationTaskClasses = deepcopy(self.oldValidationTaskClasses)
        self.testingTasks = deepcopy(self.oldTestingTasks)
        self.testingTaskClasses = deepcopy(self.oldTestingTaskClasses)

class relationExtractionData(dataGenerator):
    def __init__(self, numExamples, numFeatures, relInd, pruningThres,
                 pool = True):
        dataGenerator.__init__(self,numExamples, numFeatures, pool)
        self.relInd = relInd
        self.pruningThres = pruningThres
        self.positiveExample = None
        self.negativeExample = None
        #self.generateData()

    def getName(self):
        return "relex"

    def generateData(self):
        trainingDataFile = 'data/combGaborOur_CS_and_test'
        testDataFile = 'data/test_strict_new_feature'
        crowdDataFile = open('data/crowdData', 'r')

        allFeatures = getFeatures(trainingDataFile, self.pruningThres)

        self.responses = {}
        #examples = {}        
        for line in crowdDataFile:
            line = line.split('\t')
            numTokens = len(line)
            if numTokens < 2:
                continue
            sentenceId = line[6]
            #self.responses[sentenceId] = {}
            currentTokenIndex = 7
            #Should I laplace smooth here?
            #self.responses[sentenceId] = []
            self.responses[sentenceId] = [0, 1]

            while currentTokenIndex < numTokens -1:
                #print currentTokenIndex
                #print numTokens
                workerId = line[currentTokenIndex]
                workerResponse = line[currentTokenIndex + 1].split(',')
                workerResponse = workerResponse[self.relInd]
                if 'neg' in workerResponse:
                    self.responses[sentenceId].append(0)
                else:
                    self.responses[sentenceId].append(1)                    
                currentTokenIndex += 2

        (labels, examples, exampleIds) = getTrainingData_2(
            trainingDataFile, self.relInd, allFeatures, self.responses)
        (testlabels, testexamples) = getTestAndGoldData(
            testDataFile, allFeatures, self.relInd)

        print exampleIds.keys()[0:2]
        examples = examples.toarray()
        testexamples = testexamples.toarray()

        #data = zip(examples, labels)
        shuffle(examples)
        print "Examples shuffled"
        #unshuffledData = zip(*data)
 
        self.oldTrainingTasks=[]

        i = 0
        for example in examples:
            if i % 1000 == 0:
                print i
            self.oldTrainingTasks.append(tuple(example))
            i += 1
        self.oldTrainingTaskClasses = {}
        for trainingTask in self.oldTrainingTasks:
            if trainingTask in self.oldTrainingTaskClasses:
                continue
            self.oldTrainingTaskClasses[trainingTask] = []
            for exampleId in exampleIds[trainingTask]:
                self.oldTrainingTaskClasses[
                    trainingTask] += self.responses[exampleId]
        
        self.oldTrainingTaskDifficulties = [
            0.5 for i in range(self.numTrainingTasks)]

        self.oldValidationTasks = []
        self.oldValidationTaskClasses = []
        
        self.oldTestingTasks = []
        for testexample in testexamples:
            self.oldTestingTasks.append(tuple(testexample))
        self.oldTestingTaskClasses = testlabels

        print "Duplicating the data"
        self.generateDuplicateData()

        print "Data Ready"

    def replenish(self):
        #Real data can't be replenished
        pass

    
class realData(dataGenerator):
    def __init__(self, numExamples, numFeatures, 
                 realDataFile, realDataName, pool = True):
        dataGenerator.__init__(self,numExamples, numFeatures, pool)
        self.realDataFile = realDataFile
        self.realDataName = realDataName
        self.generateData()

    def getName(self):
        return self.realDataName

    def generateData(self):

        numFeatures = self.numFeatures
        instances = []
        classes = []
        numberTrue = 0

        self.realDataFile.seek(0)
        for line in self.realDataFile:
            line = line.split(',')
            instances.append(
                tuple([float(line[i]) for i in range(numFeatures)]))
            classes.append(int(line[numFeatures]))
            
            if int(line[numFeatures]) == 1:
                numberTrue += 1
                
        print numberTrue
        data = zip(instances, classes)
        shuffle(data)
        unshuffledData = zip(*data)
        self.splitData(list(unshuffledData[0]), list(unshuffledData[1]))        

    def replenish(self):
        #Real data can't be replenished
        pass

class gaussianData(dataGenerator):
    def __init__(self, numExamples, numFeatures, pool = True):
        dataGenerator.__init__(self,numExamples, numFeatures, pool)
        self.generateData()

    def getName(self):
        return 'g7R'

    def generateData(self):
        center2 = []
        center1 = []
        numFeatures = self.numFeatures
        numExamples = self.numExamples

        for i in range(numFeatures):
            r = choice([-1,1]) * random()
            center1.append(r)
        for i in range(numFeatures):
            r = choice([-1,1]) * random()
            center2.append(r)

        cov1 = np.random.random((numFeatures, numFeatures))
        cov1 = cov1 * cov1.transpose()
        cov2 = np.random.random((numFeatures, numFeatures))
        cov2 = cov2 * cov2.transpose()

        self.center1 = np.array(center1)
        self.center2 = np.array(center2)

        self.cov1 = cov1
        self.cov2 = cov2

        numTrues = 0.0
        numFalses = 0.0
        examples = []
        dps = []
        formatString = ''

        instances = []
        classes = []

        for i in range(int(numExamples / 2)):
            features = np.random.multivariate_normal(self.center1, cov1) 
            instances.append(tuple(features))
            classes.append(1)
            numTrues += 1
        
        for i in range(int(numExamples / 2)):
            features = np.random.multivariate_normal(self.center2, cov2)   
            instances.append(tuple(features))
            classes.append(0)
            numFalses += 1

        data = zip(instances, classes)
        shuffle(data)
        unshuffledData = zip(*data)
        self.splitData(list(unshuffledData[0]), list(unshuffledData[1]))        

    def replenish(self):
        if random() < 0.5:
            features = np.random.multivariate_normal(self.center1, self.cov1) 
            task = tuple(features)
            self.trainingTasks.append(task)
            self.trainingTaskClasses[task] = 1
        else:
            features = np.random.multivariate_normal(self.center2, self.cov2)
            task = tuple(features)
            self.trainingTasks.append(task)
            self.trainingTaskClasses[task] = 0



class uniformData(dataGenerator):
    def __init__(self, numExamples, numFeatures, pool = True):
        dataGenerator.__init__(self,numExamples, numFeatures, pool)
        self.generateData()

    def getName(self):
        return 'uinf7'

    def generateData(self):
        numTrues = 0.0
        numFalses = 0.0

        formatString = ''
        for i in range(self.numFeatures):
            formatString += '%f,'
        formatString += '%d\n'

        while (numTrues < int(0.4 * self.numExamples) or 
               numTrues > int(0.6 * self.numExamples)):
            numTrues = 0.0
            numFalses = 0.0
            instances = []
            classes = []
            weights = []
            for i in range(self.numFeatures):
                r = choice([-1,1]) * random()
                weights.append(r)        
            intercept = choice([-1,1]) * random()
    
            self.weights = weights
            self.intercept = intercept

            for i in range(self.numExamples):
                features = []
                for j in range(self.numFeatures):
                    features.append(random())

                dp = 0
                for weight,feature in zip(weights, features):
                    dp += weight * feature
                if dp >= intercept:
                    c = 1
                    numTrues += 1
                    instances.append(tuple(features))
                    classes.append(c)
                elif dp < intercept:
                    c = 0
                    numFalses += 1
                    instances.append(tuple(features))
                    classes.append(c)
            
                    
        self.splitData(instances, classes)
        
    def replenish(self):
        weights = self.weights
        intercept = self.intercept
        
        features = []
        for j in range(self.numFeatures):
            features.append(random())
                
        dp = 0
        for weight,feature in zip(weights, features):
            dp += weight * feature
        task = tuple(features)
        if dp >= intercept:
            self.trainingTasks.append(task)
            self.trainingTaskClasses[task] = 1
        elif dp < intercept:
            self.trainingTasks.append(task)
            self.trainingTaskClasses[task] = 0

                
