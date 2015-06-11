from math import log
from random import sample

#the tasks must be only the ones for which we have zero labels so far
def getMostUncertainTask(classifier, tasks, taskIndices):
    
    probabilities = classifier.predict_proba(tasks)

    #print probabilities
    highestEntropy = -21930123123
    highestEntropyDistribution = None
    mostUncertainTasks = []

    for (i, probs) in zip(taskIndices, probabilities):
        entropy = 0
        for p in probs:
            entropy += p * log(p)
        entropy *= -1
    
        if entropy > highestEntropy:
            mostUncertainTasks = [i]
            highestEntropy = entropy
            highestEntropyDistribution = probs
        elif entropy == highestEntropy:
            mostUncertainTasks.append(i)
    
    return (highestEntropyDistribution, 
            sample(mostUncertainTasks, 1)[0])
