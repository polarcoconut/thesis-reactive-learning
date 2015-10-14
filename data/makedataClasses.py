from random import randint, random, choice, seed, sample, shuffle
from numpy.linalg import norm
import numpy as np
from math import ceil, floor, log
from copy import deepcopy
from angli1 import getFeatures, getTestAndGoldData, getTrainingData_2, trainAndTestModel_2
from angli2 import *
import difflib
from string import printable
from logisticRegression import LRWrapper
from scipy.stats import spearmanr, pearsonr

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


class galaxyZooData(dataGenerator):
    def __init__(self, numExamples, numFeatures, classInd, 
                 usePerfect =  False, sdss = True,  pool = True):
        dataGenerator.__init__(self,numExamples, numFeatures, pool)
        self.classInd = classInd
        self.sdss = sdss
        self.usePerfect = usePerfect

    def getName(self):
        if self.usePerfect:
            if self.sdss:
                return "gz1SDSSP-r%d" % (self.classInd)
            else:
                return "gz1P-r%d" % (self.classInd)

        else:
            if self.sdss:
                return "gz1SDSS-r%d" % (self.classInd)
            else:
                return "gz1-r%d" % (self.classInd)

    #indices 2-453 are the features
    def generateData(self):
        #galaxyData = open('data/gz1/galaxyLog1CleanSmall.csv', 'r')
        #workerData = open('data/gz1/workerLog1CleanSmall.csv', 'r')

        #galaxyData = open('data/gz1/galaxyLog1CleanMedium.csv', 'r')
        #workerData = open('data/gz1/workerLog1CleanMedium.csv', 'r')

        if self.sdss:
            #galaxyData = open('data/gz1/SDSSCleanLarge.csv', 'r')
            #workerData = open('data/gz1/workerLog1CleanLargeSDSS.csv', 'r')

            galaxyData = open('data/gz1/SDSSCleanSmall.csv', 'r')
            workerData = open('data/gz1/workerLog1CleanSmallSDSS.csv', 'r')
        else:
            galaxyData = open('data/gz1/galaxyLog1CleanLarge.csv', 'r')
            workerData = open('data/gz1/workerLog1CleanLarge.csv', 'r')

        galaxyClassesData = open('data/gz1/GalaxyZoo1_DR_table2.csv', 'r')

        firstLine = True
        exampleFeatures = {}
        exampleClasses = {}
        exampleClean = {}
        numCleanExamples = 0.0
        classDistribution = [0,0,0,0,0,0]
        print "Reading Galaxy Data"
        #There are 886K lines
        i = 0
        for line in galaxyData:
            if i % 1000 == 0:
                print i
            i+=1
            if firstLine:
                firstLine = False
                continue
            line = line.split(',')
            
            #For GZ1 features
            if self.sdss:
                features = []
                for feature in line[1:]:
                    #print feature
                    if feature == '':
                        feature = 0.0
                    features.append(float(feature))
                exampleFeatures[line[0]] = features
                #print "Num of Features"
                #print len(features)
            else:
                features = []
                for j in range(1, 454):
                    features.append(float(line[j]))
                exampleFeatures[line[0]] = features

            """
            exampleClasses[line[0]] = int(line[454 + self.classInd])
            print line[460:464]
            if int(line[463]) == 1:
                exampleClean[line[0]] = 1
                numCleanExamples += 1
            else:
                exampleClean[line[0]] = 0

            for c in range(0, 6):
                if int(line[454+c]) == 1:
                    classDistribution[c] += 1
            """
            
        #13 is spiral 
        #14 is elliptical
        firstLine = True
        for line in galaxyClassesData:
            if firstLine:
                firstLine = False
                continue
            line = line.split(',')
            if self.classInd == 0:
                exampleClasses[line[0]] = int(line[14])
            elif self.classInd == 1:
                exampleClasses[line[0]] = int(line[13])                
                
            
        #print classDistribution
        #print "Number of Clean Examples"
        #print numCleanExamples

        print "Reading Worker Data"
        #There are 34 million lines
        firstLine = True
        exampleLabels = {}
        i = 0
        for line in workerData:
            if i % 100000 == 0:
                print i
            i+=1
            if firstLine:
                firstLine = False
                continue
            line = line.split(',')
            if line[1] not in exampleLabels:
                exampleLabels[line[1]] = []
            if self.classInd == 0:
                if (int(line[2]) - 1) == self.classInd:
                    exampleLabels[line[1]].append(1)
                else:
                    exampleLabels[line[1]].append(0)
            elif self.classInd == 1:
                if (int(line[2])==2 or int(line[2])==3 or int(line[2])==4):
                    exampleLabels[line[1]].append(1)
                else:
                    exampleLabels[line[1]].append(0)
        
        print "Shuffling and picking a random sample of the data"
        print len(exampleLabels.keys())
        exampleIds = sample(exampleLabels.keys(), self.numExamples)
        print len(exampleIds)
        #print exampleIds[0:2]
        shuffle(exampleIds)

        print "Splitting the data and computing average accuracy of workers"

        self.oldTrainingTasks=[]
        self.oldTrainingTaskClasses = {}
        self.oldTestingTasks=[]
        self.oldTestingTaskClasses = []
        i = 0
        averageAccuracy = 0.0
        numTrainingExamples = int(floor(0.7*len(exampleIds)))
        for exampleId in exampleIds[0:numTrainingExamples]:
            if i % 1000 == 0:
                print i
            example = tuple(exampleFeatures[exampleId])
            self.oldTrainingTasks.append(example)
            if self.usePerfect:
                self.oldTrainingTaskClasses[example] = exampleClasses[exampleId]
            else:
                correctClass = exampleClasses[exampleId]
                accuracy = 0.0
                for label in exampleLabels[exampleId]:
                    if label == correctClass:
                        accuracy += 1.0
                accuracy /= len(exampleLabels[exampleId])
                averageAccuracy += accuracy
                self.oldTrainingTaskClasses[example] = exampleLabels[exampleId]
            i += 1
        averageAccuracy /= numTrainingExamples
        print "The average worker accuracy is"
        print averageAccuracy

        print "Checking to see if there is any correlation between worker accuracy and distances from the hyperplane"
        classifier = LRWrapper()
        classifier.retrain(
            self.oldTrainingTasks, 
            [self.oldTrainingTaskClasses[example] for example in self.oldTrainingTasks], 
            None)
        print "Trained classifier using all the gold data"
        uncertainties = []
        accuracies = []
        for exampleId in exampleIds[0:numTrainingExamples]:
            correctClass = exampleClasses[exampleId]
            accuracy = 0.0
            for label in exampleLabels[exampleId]:
                if label == correctClass:
                    accuracy += 1.0
            accuracy /= len(exampleLabels[exampleId])
            uncertainties.append(
                classifier.getUncertainty(exampleFeatures[exampleId]))
            accuracies.append(accuracy)

        print "Spearman"
        print spearmanr(uncertainties, accuracies)
        print "Pearson"
        print pearsonr(uncertainties, accuracies)

        print "Computing Skew"
        skew = 0.0
        for exampleId in exampleIds[numTrainingExamples:]:
            if i % 1000 == 0:
                print i
            example = tuple(exampleFeatures[exampleId])
            self.oldTestingTasks.append(example)
            self.oldTestingTaskClasses.append(exampleClasses[exampleId])
            if exampleClasses[exampleId] == 1:
                skew += 1.0
            i += 1

        print "The Skew of the Testing Data is"
        print skew / (len(exampleIds) - numTrainingExamples)

        self.oldTrainingTaskDifficulties = [
            0.5 for i in range(self.numTrainingTasks)]

        self.oldValidationTasks = []
        self.oldValidationTaskClasses = []
        
        print self.oldTestingTaskClasses[0:2]
        print len(self.oldTestingTaskClasses)
        print "Duplicating the data"
        self.generateDuplicateData()

        print "Data Ready"

    def replenish(self):
        #Real data can't be replenished
        pass

    
class relationExtractionData(dataGenerator):
    def __init__(self, numExamples, numFeatures, relInd, pruningThres,
                 balanceClasses = False,
                 pool = True, usePerfect = False):
        dataGenerator.__init__(self,numExamples, numFeatures, pool)
        self.relInd = relInd
        self.pruningThres = pruningThres
        self.positiveExample = None
        self.negativeExample = None
        self.balanceClasses = balanceClasses
        self.usePerfect = usePerfect
        #self.generateData()

    def getName(self):
        if self.balanceClasses:
            if self.usePerfect:
                return "relexBP-t%d-r%d" % (self.pruningThres, self.relInd)
            else:
                return "relexB-t%d-r%d" % (self.pruningThres, self.relInd)
        else:
            if self.usePerfect:
                return "relexP-t%d-r%d" % (self.pruningThres, self.relInd)
            else:
                return "relex-t%d-r%d" % (self.pruningThres, self.relInd)

    def generateData(self):
        trainingDataFile = 'data/relex/chris_data_exp_2000'
        crowdDataFile = open('data/relex/crowd_data_2000', 'r')

        #trainingDataFile = 'data/combGaborOur_CS_and_test'
        #crowdDataFile = open('data/crowdData', 'r')

        testDataFile = 'data/test_strict_new_feature'


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

            #If I wanted to do laplace smoothing, I would do that here.
            self.responses[sentenceId] = []

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

        #print exampleIds.keys()[0:2]
        examples = examples.toarray()
        testexamples = testexamples.toarray()

        #data = zip(examples, labels)
        examples = sample(examples, self.numExamples)
        shuffle(examples)
        print "Examples shuffled"
        print "Total number of examples"
        print len(examples)
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
                

        """
        print "Computing the entropy in each feature"
        numFeatures = len(self.oldTrainingTasks[0])
        averageEntropy = 0.0
        for i in range(numFeatures):
            entropy0 = 1.0
            entropy1 = 1.0
            for trainingTask in self.oldTrainingTasks:
                if trainingTask[i] == 0:
                    entropy0 += 1
                else:
                    entropy1 += 1
            normalizer = entropy0 + entropy1
            entropy0 = entropy0 / normalizer
            entropy1 = entropy1 / normalizer
            entropy = -1* ((entropy0 * log(entropy0)) + 
                           (entropy1 * log(entropy1)))
            averageEntropy += entropy
            if entropy1 > 0.5:
                print (entropy0, entropy1, entropy)
        averageEntropy /= numFeatures
        print averageEntropy
        """

        if self.balanceClasses:        
            print "Computing Skew in Training Data"
            numPositiveExamples = 0.0
            numNegativeExamples = 0.0
            classMajorityVotes = {}
            for trainingTask in self.oldTrainingTasks:
                counter = 0
                for label in self.oldTrainingTaskClasses[trainingTask]:
                    if label == 1:
                        counter += 1
                    else:
                        counter -= 1
                if counter > 0:
                    numPositiveExamples += 1
                    classMajorityVotes[trainingTask] = 1
                else:
                    numNegativeExamples += 1
                    classMajorityVotes[trainingTask] = 0

            print numPositiveExamples / len(self.oldTrainingTasks)
            print numNegativeExamples / len(self.oldTrainingTasks)

            print "Balancing the Training Set"
            newNumPositiveExamples = 0.0
            newNumNegativeExamples = 0.0
            newTrainingTasks = []
            for trainingTask in self.oldTrainingTasks:
                if (newNumPositiveExamples == numPositiveExamples and
                    newNumNegativeExamples == numNegativeExamples):
                    break
                if (classMajorityVotes[trainingTask] == 1  and
                    newNumPositiveExamples < numPositiveExamples):
                    newTrainingTasks.append(trainingTask)
                    newNumPositiveExamples += 1
                if (classMajorityVotes[trainingTask] == 0  and
                    newNumNegativeExamples < numPositiveExamples):
                    newTrainingTasks.append(trainingTask)
                    newNumNegativeExamples += 1

            self.oldTrainingTasks = newTrainingTasks

            print newNumPositiveExamples
            print newNumNegativeExamples
            print "Done balancing the training set"
            print "The size of the training set is"
            print len(self.oldTrainingTasks)

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

                
class triGaussianData(dataGenerator):
    def __init__(self, numExamples, numFeatures, pool = True):
        dataGenerator.__init__(self,numExamples, numFeatures, pool)
        self.generateData()

    def getName(self):
        return '3g7R'

    def generateData(self):
        center3 = []
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
        for i in range(numFeatures):
            r = choice([-1,1]) * random()
            center3.append(r)



        cov1 = np.random.random((numFeatures, numFeatures))
        cov1 = cov1 * cov1.transpose()
        cov2 = np.random.random((numFeatures, numFeatures))
        cov2 = cov2 * cov2.transpose()
        cov3 = np.random.random((numFeatures, numFeatures))
        cov3 = cov3 * cov3.transpose()


        self.center1 = np.array(center1)
        self.center2 = np.array(center2)
        self.center3 = np.array(center3)

        self.cov1 = cov1
        self.cov2 = cov2
        self.cov3 = cov3

        numTrues = 0.0
        numFalses = 0.0
        examples = []
        dps = []
        formatString = ''

        instances = []
        classes = []

        for i in range(int(numExamples / 4)):
            features = np.random.multivariate_normal(self.center1, cov1) 
            instances.append(tuple(features))
            classes.append(1)
            numTrues += 1
        
        for i in range(int(numExamples / 2)):
            features = np.random.multivariate_normal(self.center2, cov2)   
            instances.append(tuple(features))
            classes.append(0)
            numFalses += 1

        for i in range(int(numExamples / 4)):
            features = np.random.multivariate_normal(self.center3, cov3)   
            instances.append(tuple(features))
            classes.append(0)
            numFalses += 1

        data = zip(instances, classes)
        shuffle(data)
        unshuffledData = zip(*data)
        self.splitData(list(unshuffledData[0]), list(unshuffledData[1]))        

    def replenish(self):
        if random() < 0.25:
            features = np.random.multivariate_normal(self.center1, self.cov1) 
            task = tuple(features)
            self.trainingTasks.append(task)
            self.trainingTaskClasses[task] = 1
        else:
            if random() < 0.5:
                features = np.random.multivariate_normal(self.center2, 
                                                         self.cov2)
                task = tuple(features)
                self.trainingTasks.append(task)
                self.trainingTaskClasses[task] = 0
            else:
                features = np.random.multivariate_normal(self.center3, 
                                                         self.cov3)
                task = tuple(features)
                self.trainingTasks.append(task)
                self.trainingTaskClasses[task] = 0

