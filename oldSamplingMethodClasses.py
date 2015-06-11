#THIS METHOD RELABELS THE SAME THING OVER AND OVER
class impactSamplingRE(samplingMethod):
    def __init__(self):
        self.lastLabelIndex = None

    def reinit(self):
        self.__init__()

    #impactNM means that we evaluate relabeling by not assuming we 
    #pick the last item labeled, but by uncertainty sampling
    def getName(self):
        return 'impactPriorRE'

    def sample(self,activeTaskIndices, tasks,
        validationTasks, state, classifier, accuracy):

        nonactiveTaskIndices = set(range(len(tasks))).difference(
            set(activeTaskIndices))

        (p, ALIndex) = getMostUncertainTask(tasks, 
                                            activeTaskIndices, classifier)

        (p, RLIndex) = getMostUncertainTask(tasks, nonactiveTaskIndices,
                                            classifier)

        if self.lastLabelIndex == None:
            self.lastLabelIndex = ALIndex
            return ALIndex

        #changeFromRelabeling = getChangeInClassifier(
        #    tasks, state, classifier, accuracy, self.lastLabelIndex)
        changeFromRelabeling = getChangeInClassifierRE(
            tasks, state, classifier, accuracy, RLIndex)

        changeFromAL = getChangeInClassifier(
            tasks, state, classifier, accuracy, ALIndex)

        #print "IMPACT SAMPLING"
        #print RLIndex
        #print state[RLIndex]
        #print changeFromRelabeling
        #print changeFromAL

        if changeFromRelabeling > changeFromAL:
            #return self.lastLabelIndex
            return RLIndex
        else:
            self.lastLabelIndex = ALIndex
            return ALIndex



class impactSamplingExpectedExpectedMax(samplingMethod):
    def __init__(self):
        self.lastLabelIndex = None
        self.zeroImpactALIndices = set([])
        self.zeroImpactRLIndices = set([])

    def reinit(self):
        self.__init__()

    #impactNM means that we evaluate relabeling by not assuming we 
    #pick the last item labeled, but by uncertainty sampling
    def getName(self):
        return 'impactPriorExpectedExpectedMax2'

    def sample(self,activeTaskIndices, tasks,
        validationTasks, state, classifier, accuracy):

        nonactiveTaskIndices = set(range(len(tasks))).difference(
            set(activeTaskIndices).union(self.zeroImpactRLIndices))

        print "SIZES"
        print len(activeTaskIndices)
        print len(nonactiveTaskIndices)
        print len(self.zeroImpactRLIndices)
        print len(self.zeroImpactALIndices)


        (p, ALIndex) = getMostUncertainTask(
            tasks, set(activeTaskIndices).difference(self.zeroImpactALIndices), 
            classifier)

        (p, RLIndex) = getMostUncertainTask(tasks, nonactiveTaskIndices,
                                            classifier)

        if self.lastLabelIndex == None:
            self.lastLabelIndex = ALIndex
            return ALIndex

        #changeFromRelabeling = getChangeInClassifier(
        #    tasks, state, classifier, accuracy, self.lastLabelIndex)
        changeFromRelabeling = getChangeInClassifierExpectedExpectedMax(
            tasks, state, classifier, accuracy, RLIndex)

        #compareChange = getChangeInClassifier(tasks, state, 
        #                                      classifier, accuracy, RLIndex)

        changeFromAL = getChangeInClassifier(
            tasks, state, classifier, accuracy, ALIndex)

        #print "IMPACT SAMPLING"
        #print RLIndex
        #print state[RLIndex]
        #print changeFromRelabeling
        #print compareChange
        #print changeFromAL

        if changeFromAL == 0.0:
            #print "Adding index"
            #print ALIndex
            self.zeroImpactALIndices.add(ALIndex)
        if changeFromRelabeling == 0.0:
            self.zeroImpactRLIndices.add(RLIndex)

        if changeFromRelabeling > changeFromAL:
            #return self.lastLabelIndex
            return RLIndex
        else:
            self.lastLabelIndex = ALIndex
            return ALIndex


class impactSamplingExpectedMax(samplingMethod):
    def __init__(self):
        self.lastLabelIndex = None
        self.zeroImpactALIndices = set([])
        self.zeroImpactRLIndices = set([])

    def reinit(self):
        self.__init__()

    #impactNM means that we evaluate relabeling by not assuming we 
    #pick the last item labeled, but by uncertainty sampling
    def getName(self):
        return 'impactPriorExpectedMax'

    def sample(self,activeTaskIndices, tasks,
        validationTasks, state, classifier, accuracy):

        nonactiveTaskIndices = set(range(len(tasks))).difference(
            set(activeTaskIndices).union(self.zeroImpactRLIndices))

        (p, ALIndex) = getMostUncertainTask(
            tasks, set(activeTaskIndices).difference(self.zeroImpactALIndices), 
            classifier)

        (p, RLIndex) = getMostUncertainTask(tasks, nonactiveTaskIndices,
                                            classifier)

        if self.lastLabelIndex == None:
            self.lastLabelIndex = ALIndex
            return ALIndex

        #changeFromRelabeling = getChangeInClassifier(
        #    tasks, state, classifier, accuracy, self.lastLabelIndex)
        changeFromRelabeling = getChangeInClassifierExpectedMax(
            tasks, state, classifier, accuracy, RLIndex)

        compareChange = getChangeInClassifier(tasks, state, 
                                              classifier, accuracy, RLIndex)

        changeFromAL = getChangeInClassifier(
            tasks, state, classifier, accuracy, ALIndex)

        #print "IMPACT SAMPLING"
        #print RLIndex
        #print state[RLIndex]
        #print changeFromRelabeling
        #print changeFromAL
        #print compareChange

        if changeFromAL == 0.0:
            self.zeroImpactALIndices.add(ALIndex)
        if changeFromRelabeling == 0.0:
            self.zeroImpactRLIndices.add(RLIndex)

        if changeFromRelabeling > changeFromAL:
            #return self.lastLabelIndex
            return RLIndex
        else:
            self.lastLabelIndex = ALIndex
            return ALIndex


class impactSamplingMax(samplingMethod):
    def __init__(self):
        self.lastLabelIndex = None

    def reinit(self):
        self.__init__()

    #impactNM means that we evaluate relabeling by not assuming we 
    #pick the last item labeled, but by uncertainty sampling
    def getName(self):
        return 'impactPriorMax'

    def sample(self,activeTaskIndices, tasks,
        validationTasks, state, classifier, accuracy):

        nonactiveTaskIndices = set(range(len(tasks))).difference(
            set(activeTaskIndices))

        (p, ALIndex) = getMostUncertainTask(tasks, 
                                            activeTaskIndices, classifier)

        (p, RLIndex) = getMostUncertainTask(tasks, nonactiveTaskIndices,
                                            classifier)

        if self.lastLabelIndex == None:
            self.lastLabelIndex = ALIndex
            return ALIndex

        #changeFromRelabeling = getChangeInClassifier(
        #    tasks, state, classifier, accuracy, self.lastLabelIndex)
        changeFromRelabeling = getChangeInClassifierMax(
            tasks, state, classifier, accuracy, RLIndex)

        changeFromAL = getChangeInClassifier(
            tasks, state, classifier, accuracy, ALIndex)

        #print "IMPACT SAMPLING"
        #print RLIndex
        #print state[RLIndex]
        #print changeFromRelabeling
        #print changeFromAL

        if changeFromRelabeling > changeFromAL:
            #return self.lastLabelIndex
            return RLIndex
        else:
            self.lastLabelIndex = ALIndex
            return ALIndex

class cvSamplingBatch(samplingMethod):
    def __init__(self):
        self.lastAccuracy = -1.0 * float('inf')
        self.history = 1.0 #start with unilabeling 
        self.batchSize = 10
        self.currentBatchIndex = 10
        self.historyWeight = 0.1

    def reinit(self):
        self.__init__()

    def getName(self):
        return 'cvBatch'

    def sample(self,activeTaskIndices, tasks,
        validationTasks, state, classifier, accuracy):

        nonactiveTaskIndices = set(range(len(tasks))).difference(
            set(activeTaskIndices))

        #print nonactiveTaskIndices
        numFolds = 5
        sampleSize = int(floor(len(nonactiveTaskIndices) / numFolds))
        #print sampleSize
        totalAccuracy = 0.0
        if self.currentBatchIndex < self.batchSize:
            self.currentBatchIndex += 1
            if self.currentBatchIndex < floor(self.history * self.batchSize):
                (p, index) = getMostUncertainTask(tasks, activeTaskIndices, 
                                                  classifier)
                return index
            else:
                return getMostUncertainTask(tasks, nonactiveTaskIndices,
                                            classifier)[1]
        self.currentBatchIndex = 0

        for k in range(numFolds * 2):
            cvTaskIndices = sample(nonactiveTaskIndices, sampleSize)
            cvTrainingState = [[0,0] for j in range(len(tasks))]
            #cvTrainingTasks = []
            cvTrainingState.append(-1)

            cvTasks = []
            cvTaskLabels = []
        
            for nonactiveTaskIndex in nonactiveTaskIndices:
                if nonactiveTaskIndex in cvTaskIndices:
                    aggregatedLabel = np.argmax(state[nonactiveTaskIndex])
                    cvTasks.append(tasks[nonactiveTaskIndex])
                    cvTaskLabels.append(aggregatedLabel)
                else:
                    cvTrainingState[
                        nonactiveTaskIndex] = state[nonactiveTaskIndex]
                    #cvTrainingTasks.append(tasks[nonactiveTaskIndex])
            #print cvTrainingState
            retrain(tasks, cvTrainingState, 
                    classifier, True, accuracy)
            totalAccuracy += classifier.score(cvTasks, cvTaskLabels)

        averageAccuracy = totalAccuracy / numFolds

                
        if  averageAccuracy > self.lastAccuracy:
            self.history = ((self.historyWeight * self.history) +
                            (1.0-self.historyWeight) * 1.0)
            print "NEW HISTORY - UNILABEL"
            print self.history
            (p, index) = getMostUncertainTask(tasks, activeTaskIndices, 
                                              classifier)
            self.lastAccuracy = averageAccuracy
            return index
        else:
            self.history = ((self.historyWeight * self.history) +
                            (1.0-self.historyWeight) * 0.0)
            print "NEW HISTORY - RELABEL"
            print self.history
            self.lastAccuracy = averageAccuracy
            return getMostUncertainTask(tasks, nonactiveTaskIndices,
                                        classifier)[1]
            #return sample(nonactiveTaskIndices, 1)[0]

class cvSampling(samplingMethod):
    def __init__(self):
        self.lastAccuracy = -1.0 * float('inf')
        self.numFolds = 5
        self.numCVSamples = 15
    def reinit(self):
        self.__init__()

    def getName(self):
        return 'cv%d' % self.numCVSamples

    def sample(self,activeTaskIndices, tasks,
        validationTasks, state, classifier, accuracy):

        nonactiveTaskIndices = set(range(len(tasks))).difference(
            set(activeTaskIndices))

        #print nonactiveTaskIndices
        numFolds = self.numFolds
        sampleSize = int(floor(len(nonactiveTaskIndices) / numFolds))
        #print sampleSize
        totalAccuracy = 0.0
        #Do 5 fold cross validation?
        for k in range(self.numCVSamples):
            cvTaskIndices = sample(nonactiveTaskIndices, sampleSize)
            cvTrainingState = [[0,0] for j in range(len(tasks))]
            #cvTrainingTasks = []
            cvTrainingState.append(-1)

            cvTasks = []
            cvTaskLabels = []
        
            for nonactiveTaskIndex in nonactiveTaskIndices:
                if nonactiveTaskIndex in cvTaskIndices:
                    aggregatedLabel = np.argmax(state[nonactiveTaskIndex])
                    cvTasks.append(tasks[nonactiveTaskIndex])
                    cvTaskLabels.append(aggregatedLabel)
                else:
                    cvTrainingState[
                        nonactiveTaskIndex] = state[nonactiveTaskIndex]
                    #cvTrainingTasks.append(tasks[nonactiveTaskIndex])
            #print cvTrainingState
            retrain(tasks, cvTrainingState, 
                    classifier, True, accuracy)
            totalAccuracy += classifier.score(cvTasks, cvTaskLabels)

        averageAccuracy = totalAccuracy / numFolds
        if  averageAccuracy > self.lastAccuracy:
            (p, index) = getMostUncertainTask(tasks, activeTaskIndices, 
                                              classifier)
            self.lastAccuracy = averageAccuracy
            return index
        else:
            self.lastAccuracy = averageAccuracy
            return getMostUncertainTask(tasks, nonactiveTaskIndices,
                                        classifier)[1]
            #return sample(nonactiveTaskIndices, 1)[0]
    

class validationSampling(samplingMethod):
    def __init__(self):
        self.lastValidationAccuracy = None
        self.validationIncreasing = True
    
    def reinit(self):
        self.lastValidationAccuracy = None
        self.validationIncreasing = True

    def getName(self):
        return 'valid'

    def sample(self,activeTaskIndices, tasks,
        validationTasks, state, classifier, accuracy):
        if self.validationIncreasing:
            (p, index) = getMostUncertainTask(tasks, activeTaskIndices, 
                                              classifier)
            return index
        else:
            nonactiveTaskIndices = set(range(len(tasks))).difference(
                set(activeTaskIndices))
            #return getMostUncertainTask(tasks, nonactiveTaskIndices,
            #                            classifier)[1]
            return sample(nonactiveTaskIndices, 1)[0]

    def validate(self):
        return True

    def setValidation(self, validationAccuracy):
        if validationAccuracy > self.lastValidationAccuracy:
            self.validationIncreasing = False
        else:
            self.validationIncreasing = True
        self.lastValidationAccuracy = validationAccuracy


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
class simulatedAnnealing:

    def __init__(self,temp, cr):
        self.temp = temp
        self.coolingRate = cr

    def sample(self, activeTaskIndices, trainingTasks,
                           validationTasks, 
                           state, classifier,
                           accuracy):
        bestRisk = float('inf')
        bestTaskIndex = None
        bestTask = None

        #print classifier

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
                self.temp *= (1.0 - self.coolingRate)
                return neighborIndex
            else:
                if random() < exp((oldRisk - totalRisk) / self.temp):
                    self.temp *= (1.0 - self.coolingRate)
                    return neighborIndex
                self.temp *= (1.0 - self.coolingRate)

        return bestTaskIndex


class impactSamplingMedium(samplingMethod):
    def __init__(self, strategies, parameters):
        self.baseStrategies = strategies
        self.outputString = ""
        filename = "outputs/g7/" + self.getName() 
        filename += "-f%d-lr-g%.1f-%d-%d-impactStats-10"
        self.logFile = open(filename % parameters, 'w')
        
    def reinit(self):
        for baseStrategy in self.baseStrategies:
            baseStrategy.reinit()
        self.logFile.write(self.outputString)
        self.logFile.write("\n")
        self.outputString = ""

    #impactNM means that we evaluate relabeling by not assuming we 
    #pick the last item labeled, but by uncertainty sampling
    def getName(self):
        strategiesString = ""
        for strategy in self.baseStrategies:
            strategiesString += strategy.getName()
        return 'impactPriorNeighbor%s' % strategiesString

    def sample(self,activeTaskIndices, tasks,
        validationTasks, state, classifier, accuracy):

        nonactiveTaskIndices = set(range(len(tasks))).difference(
            set(activeTaskIndices))

        candidateIndices = []
        changes = []
        bestIndices = []
        bestChange = 0
        #print "TRYING ALL STRATEGIES"
        for (strategyNumber, baseStrategy) in zip(
                range(len(self.baseStrategies)), self.baseStrategies):
            #Assuming all the base strategies are alphas, we dont have to
            #deepcopy activeTaskIndices
            baseStrategyIndex = baseStrategy.sample(
                activeTaskIndices, tasks, validationTasks,
                state, classifier, accuracy)
            #print strategyNumber
            #print baseStrategyIndex
            candidateIndices.append(baseStrategyIndex)
            #print [classifier.predict_proba([tasks[ci]])[0] for ci in candidateIndices]
            #print state[baseStrategyIndex]
            baseStrategyChange = getChangeInClassifier(
                tasks, state, classifier, accuracy, baseStrategyIndex)
            if baseStrategyChange > bestChange:
                bestIndices = [(strategyNumber, baseStrategyIndex)]
                bestChange = baseStrategyChange
            elif baseStrategyChange == bestChange:
                bestIndices.append((strategyNumber, baseStrategyIndex))
            else:
                continue

        #print "IMPACT SAMPLING"
        #print RLIndex
        #print state[RLIndex]
        #print changeFromRelabeling
        #print changeFromAL

        #print "BEST INDICES"
        #print bestIndices
        (nextStrategyNumber, nextIndex) = sample(bestIndices,1)[0]
        self.outputString += ("%f\t" % nextStrategyNumber)
        return nextIndex

class impactSamplingEMMedium(samplingMethod):
    def __init__(self, strategies):
        self.baseStrategies = strategies
        self.outputString = ""
        
    def reinit(self):
        for baseStrategy in self.baseStrategies:
            baseStrategy.reinit()
        self.logFile.write(self.outputString)
        self.logFile.write("\n")
        self.outputString = ""

    #impactNM means that we evaluate relabeling by not assuming we 
    #pick the last item labeled, but by uncertainty sampling
    def getName(self):
        strategiesString = ""
        for strategy in self.baseStrategies:
            strategiesString += strategy.getName()
        return 'impactPriorEMNeighbor%s' % strategiesString

    def sample(self,activeTaskIndices, tasks,
        validationTasks, state, classifier, accuracy):

        nonactiveTaskIndices = set(range(len(tasks))).difference(
            set(activeTaskIndices))

        candidateIndices = []
        changes = []
        bestIndices = []
        bestChange = 0
        #print "TRYING ALL STRATEGIES"
        for (strategyNumber, baseStrategy) in zip(
                range(len(self.baseStrategies)), self.baseStrategies):
            #Assuming all the base strategies are alphas, we dont have to
            #deepcopy activeTaskIndices
            baseStrategyIndex = baseStrategy.sample(
                activeTaskIndices, tasks, validationTasks,
                state, classifier, accuracy)
            #print strategyNumber
            #print baseStrategyIndex
            candidateIndices.append(baseStrategyIndex)
            #print [classifier.predict_proba([tasks[ci]])[0] for ci in candidateIndices]
            #print state[baseStrategyIndex]
            if baseStrategyIndex in activeTaskIndices:                
                baseStrategyChange = getChangeInClassifier(
                    tasks, state, classifier, accuracy, baseStrategyIndex)
            else:
                baseStrategyChange = getChangeInClassifierExpectedMax(
                    tasks, state, classifier, accuracy, baseStrategyIndex)

            if baseStrategyChange > bestChange:
                bestIndices = [(strategyNumber, baseStrategyIndex)]
                bestChange = baseStrategyChange
            elif baseStrategyChange == bestChange:
                bestIndices.append((strategyNumber, baseStrategyIndex))
            else:
                continue

        #print "IMPACT SAMPLING"
        #print RLIndex
        #print state[RLIndex]
        #print changeFromRelabeling
        #print changeFromAL

        #print "BEST INDICES"
        #print bestIndices
        (nextStrategyNumber, nextIndex) = sample(bestIndices,1)[0]
        self.outputString += ("%f\t" % nextStrategyNumber)
        return nextIndex
