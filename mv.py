

class MV:

    def __init__(self):
        self.trainingData = {}

    def train(self):
        pass

    
    #task can be features, or a taskID.
    def addTrainingLabel(self, task, label):
        if task not in self.trainingData:
            self.trainingData[task] = [label]
        else:
            self.trainingData[task].append(label)


    
