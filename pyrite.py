from random import sample, randint, random
import cPickle as pickle

from utils import *
from runLotsFunction import * 

from math import ceil
import numpy as np
from itertools import product

from sklearn import tree
from sklearn.cluster import SpectralClustering
from sklearn.semi_supervised import LabelSpreading

from data.makedata import makeGaussianData
from scipy.stats import spearmanr


def doPyriteSSL(instances, budget):

    print "Doing SSL Pyrite"

    print len(instances)
    print budget

    ssl1 = LabelSpreading(kernel = 'knn')
    #ssl2 = LabelSpreading(kernel = 'rbf')

    fakeUnsupervisedLabels = [-1 for i in range(len(instances))]
    fakeUnsupervisedLabels[0] = 0
    fakeUnsupervisedLabels[1] = 1

    ssl1.fit(np.array(instances), fakeUnsupervisedLabels)
    #ssl2.fit(np.array(instances), fakeUnsupervisedLabels)
    predictedLabels1 = ssl1.predict(instances)
    #predictedLabels2 = ssl2.predict(instances)
        
    
    (predictedBestStrategy1, 
     predictedAccuracies1) = getBestPolicy2(budget = budget, 
                                            budgetInterval = budget, 
                                            workerAccuracies = [1.0],
                                            numSimulations = 1000, 
                                            instances = instances,
                                            classes = predictedLabels1,
                                            classifier = LRWrapper(),
                                            d = 0.5, numClasses = 2,
                                            policies = [0,1,2,3],
                                            writeToFile = False)
    predictedBestStrategy1 = predictedBestStrategy1[0,0]
    """
    (predictedBestStrategy2, 
     predictedAccuracies2 ) = getBestPolicy2(budget = budget, 
                                             budgetInterval = budget, 
                                             workerAccuracies = [1.0],
                                             numSimulations = 1000, 
                                             instances = instances,
                                             classes = predictedLabels2,
                                             classifier = LRWrapper(),
                                             d = 0.5, numClasses = 2,
                                             policies = [0,1,2,3],
                                             writeToFile = False)
    predictedBestStrategy2 = predictedBestStrategy2[0,0]
        
         
        
    if (max(predictedAccuracies1[(1.0, budget)].values()) >
        max(predictedAccuracies2[(1.0, budget)].values())):
        predictedBestStrategy = predictedBestStrategy1
    else:
        predictedBestStrategy = predictedBestStrategy2
    
    print "Finished Pyrite"
    print "Best Initial Strategy"
    print predictedBestStrategy
    return int(predictedBestStrategy)
    """

    predictedBestStrategy = predictedBestStrategy1

    return int(predictedBestStrategy)
def doPyrite(instances, budget):
    print "Doing Pyrite"

    print "CLUSTERING ONCE"
    sc = SpectralClustering(n_clusters=2,
                            affinity = 'nearest_neighbors')
    predictedLabels1 = sc.fit_predict(np.array(instances))
    
    print "CLUSTERING TWICE"
    sc = SpectralClustering(n_clusters=2)
    predictedLabels2 = sc.fit_predict(np.array(instances))
        
    
    print "GETTING BEST POLICY ONCE"
    (predictedBestStrategy1, 
     predictedAccuracies1) = getBestPolicy2(budget = budget, 
                                            budgetInterval = budget, 
                                            workerAccuracies = [1.0],
                                            numSimulations = 1000, 
                                            instances = instances,
                                            classes = predictedLabels1,
                                            classifier = LRWrapper(),
                                            d = 0.5, numClasses = 2,
                                            policies = [0,1,2,3],
                                            writeToFile = False)
    predictedBestStrategy1 = predictedBestStrategy1[0,0]
    
    print "GETTING BEST POLICY TWICE"
    (predictedBestStrategy2, 
     predictedAccuracies2 ) = getBestPolicy2(budget = budget, 
                                             budgetInterval = budget, 
                                             workerAccuracies = [1.0],
                                             numSimulations = 1000, 
                                             instances = instances,
                                             classes = predictedLabels2,
                                             classifier = LRWrapper(),
                                             d = 0.5, numClasses = 2,
                                             policies = [0,1,2,3],
                                             writeToFile = False)
    predictedBestStrategy2 = predictedBestStrategy2[0,0]
        
         
        
    if (max(predictedAccuracies1[(1.0, budget)].values()) >
        max(predictedAccuracies2[(1.0, budget)].values())):
        predictedBestStrategy = predictedBestStrategy1
    else:
        predictedBestStrategy = predictedBestStrategy2

    print "Finished Pyrite"
    print "Nearnest Neighbor Strategy"
    print predictedBestStrategy1
    print predictedAccuracies1
    print "RBF Strategy"
    print predictedBestStrategy2
    print predictedAccuracies2

    print "Best Initial Strategy"
    print predictedBestStrategy
    return int(predictedBestStrategy)
