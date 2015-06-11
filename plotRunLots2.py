import matplotlib.pyplot as plt 
import numpy as np
import cPickle as pickle

budget =3450
budgetInterval = 50
budgets = range(budgetInterval, budget+1, budgetInterval)
#workerAccuracies = [3.3, 1.7, 1.0, 0.5, 0.15]
workerAccuracies = [1.0]
#workerAccuracies = [3.3]
#workerAccuracies = [0.15]
#workerAccuracies = [0.0]

spotPrintBudget = 1150

averageAccuracies = pickle.load(open(
        "realdataresults/runlotsAccuracies57,4601,3450,50",         
        "rb"))

#averageAccuracies = pickle.load(open(
#        "realdataresults/runlotsAccuracies100,606,454,50", 
#        "rb"))



#averageErrors = pickle.load(open(
#        "results/runlotsErrors16", 
#        "rb"))

#xprint averageAccuracies

for (i, gamma) in zip(range(len(workerAccuracies)), workerAccuracies):
    fig = plt.figure()
    ax = fig.add_subplot(111)
 #fig, ax = plt.subplots()
    ax.set_xlabel('Budget')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0.5, 1.0])
    #ax.set_ylabel('Relabeling Accuracy / Unilabling Accuracy')
    
    accuracies0 = []
    accuracies1 = []
    accuracies2 = []
    accuracies3 = []

    errors0 = []
    errors1 = []
    errors2 = []
    errors3 = []


    ratios1 = []
    ratios2 = []
    ratios3 = []

    for (j, b) in zip(range(len(budgets)), budgets):
        accuracies0.append(averageAccuracies[(gamma, b)][0])
        accuracies1.append(averageAccuracies[(gamma, b)][1])
        accuracies2.append(averageAccuracies[(gamma, b)][2])
        accuracies3.append(averageAccuracies[(gamma, b)][3])

        if b == spotPrintBudget:
            print averageAccuracies[(gamma, b)][0]
            print averageAccuracies[(gamma, b)][1]
            print averageAccuracies[(gamma, b)][2]
            print averageAccuracies[(gamma, b)][3]

        #errors0.append(averageErrors[(gamma, b)][0])
        #errors1.append(averageErrors[(gamma, b)][1])
        #errors2.append(averageErrors[(gamma, b)][2])
        #errors3.append(averageErrors[(gamma, b)][3])

        ratios1.append(averageAccuracies[(gamma ,b)][1] /
                       averageAccuracies[(gamma, b)][0])
        ratios2.append(averageAccuracies[(gamma ,b)][2] /
                       averageAccuracies[(gamma, b)][0])
        ratios3.append(averageAccuracies[(gamma ,b)][3] /
                       averageAccuracies[(gamma, b)][0])

         
    
    plt.plot(budgets, accuracies0, color='0.8', linestyle='-', 
             label='Unilabeling', linewidth=5.0)
    plt.plot(budgets, accuracies1, color='0.6', linestyle='-', 
             label='2/3 Relabeling', linewidth=5.0)
    plt.plot(budgets, accuracies2, color='0.3', linestyle='-', 
             label='3/5 Relabeling', linewidth=5.0)
    plt.plot(budgets, accuracies3, color='0.1', linestyle='-', 
             label='4/7 Relabeling', linewidth=5.0)
    

    """
    plt.plot(budgets, ratios1, color='0.8', linestyle='-', 
             label='2/3 Relabeling', linewidth=5.0)
    plt.plot(budgets, ratios2, color='0.5', linestyle='-', 
             label='3/5 Relabeling', linewidth=5.0)
    plt.plot(budgets, ratios3, color='0.2', linestyle='-', 
             label='4/7 Relabeling', linewidth=5.0)
    """
    
    handles, labels = ax.get_legend_handles_labels()
    fig.set_facecolor('white')
    if gamma <= 0.5:
        ax.legend(handles, labels, bbox_to_anchor=(0, 0), 
                  loc=2, borderaxespad=0.0)
    else:
        ax.legend(handles, labels, bbox_to_anchor=(0, 1), 
                  loc=2, borderaxespad=0.0)



    #print accuracies0
    #print errors0
    #print accuracies1
    #print errors1
    #print accuracies2
    #print errors2
    #print accuracies3
    #print errors3

    plt.show()
    plt.clf()

