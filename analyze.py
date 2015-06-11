import numpy as np
from numpy.linalg import norm


#f = open('data/data_banknote_authentication.txt', 'r')
#f = open('data/breast-cancer-wisconsin-formatted.data', 'r')
#f = open('data/iris.data', 'r')
#f = open('data/eegeyestate.data', 'r') #14 features 
f = open('data/seismicbumps.data', 'r') #18 features


numFeatures = 18
numClasses = 2

instances = []
classes = []
numberTrue = 0

for line in f:
    line = line.split(',')
    #print line
    instances.append(np.array([float(line[i]) for i in range(numFeatures)]))
    classes.append(int(line[numFeatures]))
    
    if int(line[numFeatures]) == 1:
        numberTrue += 1

priorTrue = (1.0 * numberTrue) / len(instances)

print "Prior True:"
print priorTrue

instances = np.array(instances)
classes = np.array(classes)

#For each class, calculate the average and max l2 distance from each other
averageVectors = [np.array([0 for i in range(numFeatures)]) for j in range(numClasses + 1)]
classSizes = [0 for i in range(numClasses + 1)]
distancesWithinClasses = [0 for i in range(numClasses + 1)]

for (instance, c) in zip(instances, classes):
    averageVectors[c] += instance
    classSizes[c] += 1
    averageVectors[numClasses] += instance
    classSizes[numClasses] += 1

averageVectors = [av / cs for (av, cs) in zip(averageVectors, classSizes)]

for (instance, c) in zip(instances, classes):
    distancesWithinClasses[c] += norm(instance-averageVectors[c])
    distancesWithinClasses[numClasses] += norm(instance - averageVectors[numClasses])

distancesWithinClasses = [dc / cs for (dc, cs) in zip(distancesWithinClasses,
                                                      classSizes)]

print distancesWithinClasses
