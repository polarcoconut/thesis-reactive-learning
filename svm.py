from numpy import dot, linalg
from sklearn import svm
from sklearn.linear_model import LinearRegression
from math import log
from random import sample

class SVMWrapper:

    #k is the number of features
    def __init__(self, C = 1):
        self.k = 4
        self.C = C
    def retrain(self, examples, labels):
        #self.classifier = svm.SVC(kernel = 'poly', degree=1, coef0=0,
        #                            verbose = False)
        #self.classifier = svm.LinearSVC(C = self.C, loss='l1')
        #self.classifier = svm.SVC(C = self.C)
        #self.classifier = svm.SVC(C = 100.0)
        #self.classifier = svm.SVR(C = 10000.0)
        self.classifier = LinearRegression()
        #self.classifier = svm.SVC()
        #self.classifier = svm.SVC(kernel = 'linear', C=1e-8)
        #self.classifier = svm.LinearSVC(C = 1e-8)
        #print examples
        #print labels
        #print len(examples)
        #print len(labels)
        self.classifier.fit(examples, labels)

    def predict(self, testExamples):
        return self.classifier.predict(testExamples)

    def score(self, testExamples, labels):
        return self.classifier.score(testExamples, labels)

    def fscore(self, testExamples, labels):
        predictions = self.predict(testExamples)
        precision = 0.0
        precisionD = 0.000000001
        recall = 0.0
        recallD = 0.000000001
        for (prediction, label) in zip(predictions, labels):
            if prediction == 1:
                if label == 1:
                    precision += 1
                precisionD += 1
            if label == 1:
                if prediction == 1:
                    recall += 1
                recallD += 1
        
        precision /= precisionD
        recall /= recallD
        
        return 2 * ((precision * recall) / (precision + recall + 0.000000001))

    #distance to the hyperplane
    def getUncertainty(self, example):
        probs = self.classifier.predict_proba([example])
        entropy = 0.0
        for p in probs[0]:
            entropy += p * log(p+0.0000001)
        entropy *= -1

        return entropy

    def getAllUncertainties(self, examples):
        entropies = []
        probs = self.classifier.predict_proba(examples)
        for prob in probs:
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

    def getMostUncertainTask(self, tasks, taskIndices):
        highestUncertainty = -21930123123
        highestEntropyDistribution = None
        mostUncertainTaskIndices = []
        mustUncertainTasks = []

        entropies = self.getAllUncertainties(tasks)
        for (task, i, uncertainty) in zip(tasks, taskIndices, entropies):    
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
        
        mostUncertainTaskIndex = mostUncertainTaskIndices[0]
        mostUncertainTask = mostUncertainTasks[0]

        return (self.classifier.predict_proba([mostUncertainTask])[0], 
                mostUncertainTaskIndex)


    def getTotalUncertainty(self, examples):
        
        totalUncertainty = 0.0
        for example in examples:
            #print "YO"
            #print self.getUncertainty(example)
            totalUncertainty += self.getUncertainty(example)

        totalUncertainty /= len(examples)
        
        #return max(self.getAllUncertainties(examples))
        return totalUncertainty
