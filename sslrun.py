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
from sklearn.semi_supervised import LabelSpreading
from runLotsFunction import *
from pyrite import *

def getBestPolicySSL(budget, budgetInterval, workerAccuracies, 
                   numSimulations, instances, classes, 
                   classifier, d = 0.5, numClasses = 2,
                   policies = [0],
                   writeToFile = False):

    averageAccuracies = {}
    means = {}
    standardErrors = {}
    ratios = {}
    meanRatios = {}
    ratioSE = {}

    skips = dict((policy, []) for policy in policies)
    numExamplesUsed = dict((policy, []) for policy in policies)


    #budgets = range(budgetInterval, budget+budgetInterval, budgetInterval)
    budgets = range(budgetInterval, budget+1, budgetInterval)
    #print budgets
    for gamma in workerAccuracies:
        for b in budgets:
            averageAccuracies[(gamma,b)] = dict(
                (policy, []) for policy in policies)
            means[(gamma, b)] =  dict(
                (policy, 0) for policy in policies)
            standardErrors[(gamma, b)] =  dict(
                (policy, 0) for policy in policies)

    for simNum in range(numSimulations):

        print simNum
        
        trainingTasks = []
        trainingTaskClasses = []
        trainingTaskDifficulties = []

        validationTasks = []
        validationTaskClasses = []
        
        testingTasks = []
        testingTaskClasses = []
        
        numFeatures = len(instances[0])
        numExamples = len(instances)
        allSamples = zip(instances, classes)
        shuffle(allSamples)
        
        
        trainingSamples = allSamples[0:int(ceil(0.8 * len(allSamples)))]
        validationSamples = allSamples[int(ceil(0.8 * len(allSamples))):int(ceil(0.85 * len(allSamples)))]
        testingSamples = allSamples[int(ceil(0.85 * len(allSamples))):len(allSamples)]


        for (instance, c) in trainingSamples:
            trainingTasks.append(instance)
            trainingTaskClasses.append(c)
            trainingTaskDifficulties.append(d)
        for (instance, c) in validationSamples:
            validationTasks.append(instance)
            validationTaskClasses.append(c)
        for (instance, c) in testingSamples:
            testingTasks.append(instance)
            testingTaskClasses.append(c)
        

        state = [[0 for i in range(numClasses)] 
                 for j in range(len(trainingSamples))]
        state.append(budget)
        baselineState = deepcopy(state)

        lam = 1.0

       # print "beginning experiments"
        for gamma in workerAccuracies:
            for policy in policies:
                success = False
                numSkips = 0.0
                while not success:
                    try:
                        
                        (nExamplesUsed, accuracies) = learnSSL(
                            int(policy*2 + 1), 
                            deepcopy(baselineState), 
                            trainingTasks + validationTasks + testingTasks,
                            trainingTasks, 
                            trainingTaskDifficulties, 
                            trainingTaskClasses, 
                            testingTasks,testingTaskClasses, 
                            gamma, budget, 
                            classifier, None, None, int(budget/10), numClasses,
                            bayesOptimal = False,
                            smartBudgeting = True,
                            budgetInterval = budgetInterval)


                        numExamplesUsed[policy].append(nExamplesUsed)
                        for (i, b) in zip(range(len(budgets)), budgets):
                            accuracy = accuracies[i][0]
                            averageAccuracies[(gamma, b)][policy].append(
                                accuracy)
                        success = True
                    except Exception as e:
                        sys.stdout.flush()
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(
                            exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        print e
                        numSkips += 1.0
                skips[policy].append(numSkips)

    #Get the results and figure out the winners
    bestStrategies = np.zeros((len(workerAccuracies), len(budgets)))
    for (i, gamma) in zip(range(len(workerAccuracies)), workerAccuracies):
        for (j, b) in zip(range(len(budgets)), budgets):
            for policy in policies:
                means[(gamma, b)][policy] = np.average(
                    averageAccuracies[(gamma, b)][policy])

                standardErrors[(gamma, b)][policy] = ((1.96 * np.std(
                        averageAccuracies[(gamma, b)][policy]))/
                                                      sqrt(numSimulations))
                

            #bestStrategies[i,j] = np.argmax(means[(gamma, b)])
            bestStrategies[i,j] = max(means[(gamma, b)].iterkeys(), 
                                      key = (lambda key: means[(gamma,b)][key]))


    print bestStrategies
    print means
    print standardErrors


    if writeToFile:
        pickle.dump(means, open(
                "sc-rbf-results/runlotsAccuracies%d,%d,%d,%d" % 
                (numFeatures, numExamples, budget, budgetInterval),
                "wb"))
        pickle.dump(standardErrors, open(
                "sc-rbf-results/runlotsErrors%d,%d,%d,%d" % 
                (numFeatures, numExamples, budget, budgetInterval),
                "wb"))
        pickle.dump(bestStrategies, open(
                "sc-rbf-results/runlotsStrategies%d,%d,%d,%d" % 
                (numFeatures, numExamples, budget, budgetInterval), 
                "wb"))

    return (bestStrategies, means)




def learnSSL(numRelabels, state, allTasks, 
             trainingTasks,
             trainingTaskDifficulties,
             trainingTaskClasses,
             testingTasks,
             testingTaskClasses,
             gamma, budget, 
             classifier, 
             outputfile, interval,
             sslInterval,
             numClasses, bayesOptimal = False, smartBudgeting = False,
             budgetInterval = None):

    if budgetInterval == None:
        budgetInterval = budget

    accuracy = (1.0 / 2.0)*(1.0+((1.0 - 0.5) ** gamma))

    outputString = ""
    activeTasks = []
    activeTaskIndices = []

    accuracies = []

    activeTaskIndices = range(len(trainingTasks))
    shuffle(activeTaskIndices)
    numExamples = 0.0
    totalSpentSoFar = 0.0

    goldLabels = []

    #a hack for the first point when the budgetinterval is 1
    firstPointManaged = False
    
    for idx in range(len(deepcopy(activeTaskIndices))):

        index = passive(activeTaskIndices)
        
        #Active Learning
        """
        if idx < 20:
            index = passive(activeTaskIndices)
        else:
            retrain(trainingTasks, state[0:-1], classifier, 
                    bayesOptimal, accuracy)
            index = uncertaintySampling(activeTaskIndices,
                                        trainingTasks, classifier)
        """
                                        
        activeTaskIndices.remove(index)
            
        if state[-1] <= 0:
            break
        numExamples += 1

        for r in range(numRelabels):
            correctLabel = trainingTaskClasses[index]
            incorrectLabels = [i for i in range(numClasses)]
            incorrectLabels.remove(correctLabel)
            workerLabel = simLabel(trainingTaskDifficulties[index], gamma, 
                     trainingTaskClasses[index],
                     incorrectLabels)
            state[index][workerLabel] += 1

            state[-1] -= 1
            #print state[-1]
            totalSpentSoFar += 1
            if totalSpentSoFar % budgetInterval == 0:
                if not firstPointManaged and budgetInterval == 1:
                    accuracies.append((0.5, 0.5))
                    firstPointManaged = True
                    continue

                classifier.C = 1.0 * (1.0 * budget / numExamples)
                #classifier = LRWrapper(0.01 * (1.0 * budget / numExamples))
                #classifier = LRWrapper(1.0 * (1.0 * budget / numExamples))
                #classifier = LRWrapper(100000000.0)
                #classifier = DTWrapper()
                #classifier = SVMWrapper(1.0 * (1.0 * budget / numExamples))
                #classifier = RFWrapper()
                #classifier = NNWrapper()
                #classifier = PerceptronWrapper()

                #print "RETRAINING"
                #print totalSpentSoFar
                retrain(trainingTasks, state[0:-1], classifier, 
                        bayesOptimal, accuracy)
                accuracies.append(
                    (classifier.score(testingTasks, testingTaskClasses),
                     classifier.fscore(testingTasks, testingTaskClasses)))
                #print accuracies
            if smartBudgeting:
                if state[index][workerLabel] > int(numRelabels / 2):
                    break
            if state[-1] <= 0:
                break

        if idx > 0 and idx % sslInterval == 0:
            print "Relearning Policy"
            print idx
            ssls = []
            ssl1 = LabelSpreading(kernel = 'knn')
            #ssl2 = LabelSpreading(kernel = 'rbf')
            ssls.append(ssl1)
            #ssls.append(ssl2)

            predictedLabels = []
            bestStrategies = []
            bestStrategyAccuracies = []

            for ssl in ssls:
                (pl,
                 (labeledTasks, labeledTaskLabels))= retrainSSL(
                     allTasks, state[0:-1], ssl)
                (bestStrategy,
                 strategyAccuracies)= getBestPolicy2(
                     budget = budget, 
                     budgetInterval = budget, 
                     workerAccuracies = [1.0],
                     numSimulations = 100, 
                     instances=allTasks,
                     classes = pl,
                     classifier = classifier,
                     d = 0.5, numClasses = 2,
                     writeToFile = False,
                     testingSamples = zip(labeledTasks, labeledTaskLabels))
                bestStrategies.append(bestStrategy[0,0])
                bestStrategyAccuracies.append(
                    strategyAccuracies[(1.0, budget)][bestStrategy[0,0]])

            bestStrategy = bestStrategies[np.argmax(bestStrategyAccuracies)]

            """
            (predictedLabels1,
             (labeledTasks, labeledTaskLabels))= retrainSSL(
                 allTasks, state[0:-1], ssl1)
            (predictedLabels2,
             (labeledTasks, labeledTaskLabels))= retrainSSL(
                 allTasks, state[0:-1], ssl2)
            """

            #predictedLabels2 = retrainSSL(allTasks, state[0:-1], ssl2)

            """
            accuracy = 0.0
            inverseAccuracy = 0.0
            for (label, predictedLabel) in zip(trainingTaskClasses, 
                                               predictedLabels):
                if label == predictedLabel:
                    accuracy += 1
                if not label == predictedLabel:
                    inverseAccuracy += 1
            print "Correlations:"
            print accuracy / len(predictedLabels)
            print inverseAccuracy / len(predictedLabels)
            """
            """
            (bestStrategy1,
             strategyAccuracies1)= getBestPolicy2(budget = budget, 
                                          budgetInterval = budget, 
                                          workerAccuracies = [1.0],
                                          numSimulations = 100, 
                                          instances=allTasks,
                                          classes = predictedLabels1,
                                          classifier = classifier,
                                          d = 0.5, numClasses = 2,
                                          writeToFile = False,
)
            """

            """
            (bestStrategy1,
             strategyAccuracies1)= getBestPolicy2(
                 budget = budget, 
                 budgetInterval = budget, 
                 workerAccuracies = [1.0],
                 numSimulations = 100, 
                 instances=allTasks,
                 classes = predictedLabels1,
                 classifier = classifier,
                 d = 0.5, numClasses = 2,
                 writeToFile = False,
                 testingSamples = zip(labeledTasks, labeledTaskLabels))
            bestStrategy1 = bestStrategy1[0,0]
            
            (bestStrategy2,
             strategyAccuracies2)= getBestPolicy2(
                 budget = budget, 
                 budgetInterval = budget, 
                 workerAccuracies = [1.0],
                 numSimulations = 100, 
                 instances = allTasks,
                 classes = predictedLabels2,
                 classifier = classifier,
                 d = 0.5, numClasses = 2,
                 writeToFile = False,
                 testingSamples = zip(labeledTasks, labeledTaskLabels))
            bestStrategy2 = bestStrategy2[0,0]
            """

            """
            if (max(strategyAccuracies1[(1.0, budget)].values()) >
                max(strategyAccuracies2[(1.0, budget)].values())):
                bestStrategy = bestStrategy1
            else:
                bestStrategy = bestStrategy2
            """

            numRelabels = int(bestStrategy * 2 + 1)
            #print numRelabels
    classifier.C = 1.0 * (1.0 * budget / numExamples)

    retrain(trainingTasks, state[0:-1], classifier, bayesOptimal, accuracy)
    #print classifier.getParams()
    #print labelAccuracy(trainingTasks, state[0:-1], trainingTaskClasses)

    
    #return (classifier.score(testingTasks, testingTaskClasses),
    #        classifier.fscore(testingTasks, testingTaskClasses))
    #print accuracies
    return (numExamples, accuracies)



#dataset = 'data/breast-cancer-wisconsin-formatted.data'
#dataset = 'data/data_banknote_authentication.txt'
#dataset = 'data/seismicbumps.data'
#dataset = 'data/eegeyestate.data'
dataset = 'data/sonar.data'
#dataset = 'data/wdbc-formatted.data'
#dataset = 'data/ad-formatted.data'
#dataset = 'data/gisette.data'
#dataset = 'data/hv.data'
#dataset = 'data/hvnoise.data'
#dataset = 'data/spambase.data'
#dataset = 'data/farm-ads-formatted.data'
#dataset = 'data/hv2.data'

(instances, classes) = readData(dataset)
budget = int(floor(len(instances) * 0.5))
#budget=950
#budget = 50
budgetInterval = budget
(bestStrategy, means) = getBestPolicySSL(budget = budget, 
                                budgetInterval = budgetInterval, 
                                workerAccuracies = [1.0],
                                numSimulations = 500, 
                                instances = instances,
                                classes = classes,
                                classifier = LRWrapper(),
                                d = 0.5, numClasses = 2,
                                         policies = [0],
                                writeToFile = False)

