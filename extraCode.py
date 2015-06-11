#f1 = open('outputs/bc7/us-b10-g10', 'w')
#f2 = open('outputs/bc6/random-b10-g30', 'w')

#f3 = open('outputs/fd1/al-b150-g30-2000,400,0', 'w')
#f3a = open('outputs/fd1/al-b150-g30-f-2000,400,0', 'w')

#f4 = open('outputs/bc6/usRandom-b10-g30', 'w')

#f5 = open('outputs/bc8/usX3-b30-g30', 'w')
#f5a = open('outputs/bc8/usX3-b30-g30-f', 'w')

#f6 = open('outputs/bc6/usY-b10-g30', 'w')
#f7 = open('outputs/bc6/usZ-b10-g30', 'w')

#f8 = open('outputs/bc8/usD3-b30-g30', 'w')

#f9 = open('outputs/fd1/relearn-b150-g30-2000,10,0', 'w')
#f9a = open('outputs/fd1/relearn-b150-g30-f-2000,10,0', 'w')
#f10 = open('outputs/fd1/accRelearn-b150-g30-2000,10,0', 'w')
#f10a = open('outputs/fd1/accRelearn-b150-g30-f-2000,10,0', 'w')
#f11 = open('outputs/fd1/histRelearn-b150-g30-2000,10,0', 'w')

#f12 = open('outputs/bc1/zhao3-3-b150-g30', 'w')
#f12a = open('outputs/bc1/zhao3-3-b150-g30-f', 'w')

#f13 = open('outputs/bc10/usA-b50-g30', 'w')
#f13a = open('outputs/bc10/usA-b50-g30-f', 'w')

#f14 = open('outputs/bc10/pac-b150-g30-50', 'w')
#f14a = open('outputs/bc10/pac-b150-g30-f-50', 'w')

#f15 = open('outputs/fd1/relearn3-b300-g30-d0s991,2000,100,100', 'w')
#f16 = open('outputs/fd1/relearn5-b300-g30-d0s991,2000,100,100', 'w')
#f17 = open('outputs/fd1/relearn7-b300-g30-d0s991,2000,100,100', 'w')

#f15 = open('outputs/fd1/relearn-b300-g30-ba', 'w')

#f15a = open('outputs/fd1/al-b150-g30-2000,400,0', 'w')

#f18 = open('outputs/fd1/al-b300-g30-d0s991,2000,100,100', 'w')
#f16 = open('outputs/fd1/al-b300-g30-ba', 'w')


    """
    #LEGACY CODE FOR MDPS
    #entropy based mdp
    mdp = SimpleMDP(trainingTasks, testingTasks, LRWrapper, budget, priorTrue)
    
    #validation based mdp
    mdp2 = SimpleMDP(trainingTasks, testingTasks, LRWrapper, budget, priorTrue,
                     (validationTasks, validationTaskClasses), RL=False) 
    #1-step hack
    mdp3 = SimpleMDP(trainingTasks, testingTasks, LRWrapper, budget, priorTrue,
                     (validationTasks, validationTaskClasses))
    #uctmax
    mdp4 = SimpleMDP(trainingTasks, testingTasks, LRWrapper, budget, priorTrue,
                     (validationTasks, validationTaskClasses), RL=True) 

    #uctG
    mdp5 = SimpleMDP2(trainingTasks, testingTasks, LRWrapper, budget, priorTrue,
                     RL=False) 


    planner = UCT(mdp, 1.0, 200)
    planner2 = UCT(mdp2, 1.0, 200)
    planner4 = UCT2(mdp4, 1.0, 200)
    planner5 = UCT(mdp5, 0.5, 200)
    """

    """
    for i in range(50):
        task = randint(0, len(trainingTasks) - 1)
        nextLabel = simLabel(trainingTaskDifficulties[task], gamma, 
                             trainingTaskClasses[task])

        (trues, falses) = state[task]
        if nextLabel == 1:
            state[task] = (trues+1, falses)
        else:
            state[task] = (trues, falses+1)

    #print state[0:-1]
    retrain(trainingTasks, state[0:-1], classifier)
    print "Baseline Accuracy"
    print classifier.score(testingTasks, testingTaskClasses)
    """

    """
    classifier = LRWrapper(lam * 3)
    (score, fscore) = learn(3, deepcopy(baselineState), trainingTasks, 
          trainingTaskDifficulties, trainingTaskClasses, 
          testingTasks,testingTaskClasses, gamma, budget, 
          classifier, f15, 12, numClasses)
    relearningScores1.append(score)
    relearningFScores1.append(fscore)
    """


    """
    success = False
    numSkips = 0.0
    while not success:
        try:
            #classifier = LRWrapper(lam * 3)
            (nExamplesUsed, accuracies) = learn(3, deepcopy(baselineState), trainingTasks, 
                                    trainingTaskDifficulties, trainingTaskClasses, 
                                    testingTasks,testingTaskClasses, gamma, budget, 
                                    classifier, f15, 12, numClasses, 
                                    bayesOptimal = False,
                                    smartBudgeting = True)
            success = True
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print e

            numSkips += 1.0

    relearningSkips3.append(numSkips)
    relearningScores3.append(accuracies[0][0])
    #relearningFScores3.append(fscore)

    success = False
    numSkips = 0.0
    while not success:
        try:
            #classifier = LRWrapper(lam * 5)
            (nExamplesUsed, accuracies) = learn(5, deepcopy(baselineState), trainingTasks, 
                                    trainingTaskDifficulties, trainingTaskClasses, 
                                    testingTasks,testingTaskClasses, gamma, budget, 
                                    classifier, f16, 15, numClasses, bayesOptimal = False,
                                    smartBudgeting = True)
            success = True
        except Exception as e:
            #print e
            numSkips += 1.0
    relearningSkips5.append(numSkips)
    relearningScores5.append(accuracies[0][0])
    #relearningFScores5.append(fscore)

    success = False
    numSkips = 0.0
    while not success:
        try:
            #classifier = LRWrapper(lam * 7)
            (nExamplesUsed, accuracies) = learn(7, deepcopy(baselineState), trainingTasks, 
                                    trainingTaskDifficulties, trainingTaskClasses, 
                                    testingTasks,testingTaskClasses, gamma, budget, 
                                    classifier, f17, 14, numClasses, bayesOptimal = False,
                                    smartBudgeting = True)
            success = True
        except Exception as e:
            #print e

            numSkips += 1.0
    relearningSkips7.append(numSkips)
    relearningScores7.append(accuracies[0][0])
    #relearningFScores7.append(fscore)
    """

    """
    #Active Learning
    print "Active Learning"
    #anotherState = deepcopy(baselineState)
    #anotherState[-1] = anotherState[-1] / 3
    #print anotherState[-1]
    success = False
    numSkips = 0.0
    while not success:
        try:
            #classifier = LRWrapper(lam)
            (nExamplesUsed, accuracies) = learn(
                1, deepcopy(baselineState), 
                trainingTasks, 
                trainingTaskDifficulties, trainingTaskClasses,
                validationTasks,
                testingTasks,testingTaskClasses, gamma, budget, 
                classifier, f1, 12, numClasses)
            success = True
        except Exception as e:
            print e
            numSkips += 1.0
    activeLearningSkips.append(numSkips)
    activeLearningScores.append(accuracies[0][0])
    #activeLearningFScores.append(fscore)
    """



""""
#FOR READING FILES

budget = 0

for line in f:
    budget += 1
    line = line.split(',')
    #print line
    instances.append([float(line[i]) for i in range(numFeatures)])
    #instances.append([float(line[0]), float(line[1]), 
    #                  float(line[2]), float(line[3])])
    classes.append(int(line[numFeatures]))
    
    #True is 1 for banknote authentication
    if int(line[numFeatures]) == 1:
        numberTrue += 1
    #if int(line[numFeatures]) == 1:
    #    numberTrue += 1

    if int(line[numFeatures]) > 1:
        print budget
        print line
budget = floor(budget / 2.0)

#print instances
#print classes
priorTrue = (1.0 * numberTrue) / len(instances)

print "PRIOR TRUE:"
print priorTrue
"""


#f = open('data/fakeCircularData2000,100,0.data', 'r')
#f = open('data/fakeDataNoise2000,100,0.data', 'r')
#f = open('data/fakePDatas10011-20000,10,0.data', 'r')
#f = open('data/fakeLargeMarginC2Data2000,2,0.data', 'r') 
#f = open('data/fakeFarApartC2s1.0Data2000,1000,0.data', 'r') 
#f = open('data/fakeIntervalNoiseC2Data2000,100,0.data', 'r')
#f = open('data/fakeQC2Data2000,100,0.data', 'r')
#f = open('data/fakeNoiseC2Data2000,100,0.data','r')
#f = open('data/fakecov01-01Gs1.0C2Data4000,200,0.data', 'r')
#f = open('data/fakecov10GC2Data2000,2,0.data', 'r')
#f = open('data/fakeContainedC2Data10000,100,0.data')

    """
    (instances, classes) = makeGaussianData(budget * 2, numFeatures, 0, 
                     noise=False, numClasses=2, skew=1.0, 
                     f = None, randomData=True,
                    writeToFile = False)
    """
    

    """
    (instances, classes) = makeTriGaussianData(budget * 2, numFeatures, 0, 
                     noise=False, numClasses=2, skew=1.0, 
                     f = None, randomData=True,
                    writeToFile = False)
    """
    
    
    """
    (instances, classes) = makeLogicalData(budget * 2, numFeatures, 0, 
                     noise=False, numClasses=2, skew=1.0, 
                     f = None, randomData=True,
                    writeToFile = False)
    """
    
    #(instances, classes) = makeUniformData(budget * 2, numFeatures, 0, 
    #                                       noise=False)


    
    """
    trainingTasks = []
    trainingTaskClasses = []
    trainingTaskDifficulties = []

    #If we did have some labeled gold data
    validationTasks = []
    validationTaskClasses = []
    
    testingTasks = []
    testingTaskClasses = []


    for (instance, c) in zip(instances, classes):
        r = random()
        if r < 0.7:
            trainingTasks.append(instance)
            trainingTaskClasses.append(c)
            trainingTaskDifficulties.append(d) #Difficulty is constant
        elif r < 0.85:
            validationTasks.append(instance)
            validationTaskClasses.append(c)
        else:
            testingTasks.append(instance)
            testingTaskClasses.append(c)
    """
