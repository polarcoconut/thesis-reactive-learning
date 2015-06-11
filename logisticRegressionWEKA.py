from numpy import dot, linalg
from math import log
from random import sample

import weka.core.jvm as jvm
from weka.datagenerators import DataGenerator
from weka.core.converters import Loader, Saver
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random

class LRWrapper:

    #k is the number of features
    def __init__(self, C):
        self.k = 4
        self.C = C
        self.trainingData = None
        self.testingData = None
        #jvm.start()
        #jvm.start(system_cp = True, packages = True)

    def retrain(self, examples, labels):


        f = open("trainingweka.arff", "w")
        f.write("@relation randomset\n")
        for j in range(len(examples[0])):
            f.write("@attribute feature%d real\n" % j)
        f.write("@attribute class {TRUE, FALSE}\n")
        f.write("@data\n")

        for (example, label) in zip(examples, labels):
            for feature in example:
                f.write("%f," % feature)
            if label == 1:
                f.write("TRUE\n")
            else:
                f.write("FALSE\n")
        f.close()

        loader = Loader(classname="weka.core.converters.ArffLoader")
                       # options=["-H", "-B", "10000"])
        self.trainingData = loader.load_file("trainingweka.arff")
        self.trainingData.set_class_index(
            self.trainingData.num_attributes() - 1)
        self.classifier = Classifier(
            classname="weka.classifiers.functions.Logistic",
            options=["-R", "%f" % (1.0/self.C)])
        self.classifier.build_classifier(self.trainingData)
        


        #self.classifier = LogisticRegression(penalty = 'l2', C = self.C)
        #self.classifier = LogisticRegression()
        #self.classifier.fit(examples, labels)

    def predict(self, testExamples):
        return self.classifier.predict(testExamples)


    def getParams(self):
        return (self.classifier.coef_, self.classifier.intercept_)

    def score(self, testExamples, labels):
        f = open("testingweka.arff", "w")
        f.write("@relation randomset\n")
        for j in range(len(testExamples[0])):
            f.write("@attribute feature%d real\n" % j)
        f.write("@attribute class {TRUE, FALSE}\n")
        f.write("@data\n")
        for (example, label) in zip(testExamples, labels):
            for feature in example:
                f.write("%f," % feature)
            if label == 1:
                f.write("TRUE\n")
            else:
                f.write("FALSE\n")
        f.close()

        loader = Loader(classname="weka.core.converters.ArffLoader")
#                        options=["-H", "-B", "10000"])
        self.testingData = loader.load_file("testingweka.arff")
        self.testingData.set_class_index(self.testingData.num_attributes() - 1)

        evaluation = Evaluation(self.trainingData)
        evaluation.test_model(self.classifier, self.testingData)


        #print evaluation.percent_correct()
        #jvm.stop()
        return evaluation.percent_correct()


    def fscore(self, testExamples, labels):
        return 0
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
