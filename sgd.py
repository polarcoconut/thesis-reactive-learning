from numpy import dot, linalg
from sklearn.linear_model import SGDClassifier 
from math import log
from random import sample

class SGDWrapper:

    #k is the number of features
    def __init__(self, C = 1):
        self.k = 4
        self.C = C #This is not actually used

    #WHEN RETRAIN IS CALLED, IT'S OFFLINE LEARNING
    def retrain(self, examples, labels, weights):
        self.classifier = SGDClassifier(loss='log', penalty='l2')
        self.classifier.alpha = self.C
        #print len(examples)
        #print len(weights)
        #print "HUH"
        self.classifier.fit(examples, labels,
                            sample_weight = weights)

    #WHEN UPDATE IS CALLED, IT'S ONLINE LEARNING
    def update(self, examples, labels, weights):
        #print "UPDATING"
        self.classifier.alpha = self.C
        self.classifier.partial_fit(examples, labels, 
                                    classes=[0,1],
                                    sample_weight = weights)

    def predict(self, testExamples):
        return self.classifier.predict(testExamples)

    def predict_proba(self, testExamples):
        return self.classifier.predict_proba(testExamples)

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

