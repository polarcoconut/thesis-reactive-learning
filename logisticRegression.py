from numpy import dot, linalg
from sklearn.linear_model import LogisticRegression
from math import log
from random import sample

class LRWrapper:

    #k is the number of features
    def __init__(self, C = 1):
        self.k = 4
        self.C = C
        self.classifier = None

    def retrain(self, examples, labels, weights):
        self.classifier = LogisticRegression(penalty = 'l2', C = self.C)
        #self.classifier = LogisticRegression()
        self.classifier.fit(examples, labels)

    def predict(self, testExamples):
        return self.classifier.predict(testExamples)

    def predict_proba(self, testExamples):
        return self.classifier.predict_proba(testExamples)


    def getParams(self):
        return (self.classifier.coef_, self.classifier.intercept_)

    def score(self, testExamples, labels):
        return self.classifier.score(testExamples, labels)

    def fscore(self, testExamples, labels):
        predictions = self.predict(testExamples)
        precision = 0.0
        precisionD = 0.0
        recall = 0.0
        recallD = 0.0
        for (prediction, label) in zip(predictions, labels):
            if prediction == 1:
                if label == 1:
                    precision += 1
                precisionD += 1
            if label == 1:
                if prediction == 1:
                    recall += 1
                recallD += 1

        if precision == 0 and recall == 0:
            return  (precision, recall, 0)
        
        precision /= precisionD
        recall /= recallD

        return (precision, recall, 2 * ((precision * recall) / (precision + recall)))

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

    def getMostUncertainTask(self, tasks, activeTaskIndices):
        highestUncertainty = -21930123123
        highestEntropyDistribution = None
        mostUncertainTaskIndices = []
        mustUncertainTasks = []

        entropies = self.getAllUncertainties(tasks)
        #print entropies
        for (i, uncertainty) in zip(activeTaskIndices, entropies):
            task = tasks[i]
            if uncertainty > highestUncertainty:
                mostUncertainTaskIndices = [i]
                mostUncertainTasks = [task]
                highestUncertainty = uncertainty
            elif uncertainty == highestUncertainty:
                mostUncertainTaskIndices.append(i)
                mostUncertainTasks.append(task)

        #print "HERE"
        #(mostUncertainTaskIndex, 
        # mostUncertainTask) = sample(zip(mostUncertainTaskIndices,
        #                               mostUncertainTasks), 1)[0]
        
        mostUncertainTaskIndex = mostUncertainTaskIndices[0]
        mostUncertainTask = mostUncertainTasks[0]
        #print "THERE"
        #return mostUncertainTaskIndex
        return (self.classifier.predict_proba([mostUncertainTask])[0], 
                mostUncertainTaskIndex)

    def getExpectedError(self, tasks, activeTaskindices):
        for i in activeTaskindices:
            continue
        return None

    def getTotalUncertainty(self, examples):
        
        totalUncertainty = 0.0
        for example in examples:
            #print "YO"
            #print self.getUncertainty(example)
            totalUncertainty += self.getUncertainty(example)

        totalUncertainty /= len(examples)
        
        #return max(self.getAllUncertainties(examples))
        return totalUncertainty
