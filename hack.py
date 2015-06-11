

def getLookaheadValue(state, action):
    
        
def evalClassifier(state):
    #If we're doing active learning, take the max
    currentClassifier = self.classifier()
    if action == 0:
        retrain(self.tasks, state[0:-1], currentClassifier)
        if self.validationTasks == None:
            return currentClassifier.getTotalUncertainty(self.testingTasks)
        else:
            return -1.0 * currentClassifier.score(self.validationTasks[0],
                                                  self.validationTasks[1])
    else: #If we're doing relearning, take the average
        retrain(self.tasks, state[0:-1], currentClassifier)
        if self.validationTasks == None:
            return currentClassifier.getTotalUncertainty(self.testingTasks)
        else:
            return -1.0 * currentClassifier.score(self.validationTasks[0],
                                                  self.validationTasks[1])
