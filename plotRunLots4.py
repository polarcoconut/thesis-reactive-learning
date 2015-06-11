import matplotlib.pyplot as plt 
import numpy as np
import cPickle as pickle

budget = 20
budgetInterval = 1
budgets = range(budgetInterval, budget+budgetInterval, budgetInterval)
workerAccuracies = [3.3, 1.7, 1.0, 0.5, 0.15]
workerPercents = ['55', '65', '75', '85', '95']
workerAccuracies = [1.0]
workerPercents = ['75']
colors = ['0.9', '0.7', '0.5', '0.3', '0.1']
#workerAccuracies = [1.0]
#workerAccuracies = [0.0]

averageAccuracies = pickle.load(open(
        "results/runlotsAccuracies2,25,20,1", 
        "rb"))


fig = plt.figure()
ax = fig.add_subplot(111)
 #fig, ax = plt.subplots()
ax.set_xlabel('Budget')
ax.set_ylabel('Accuracy')
ax.set_ylim([0.5, 1.0])

for (i, gamma, p, c) in zip(range(len(workerAccuracies)), workerAccuracies,
                            workerPercents, colors):

    print i
    #ax.set_ylabel('Relabeling Accuracy / Unilabling Accuracy')
    
    accuracies0 = []

    for (j, b) in zip(range(len(budgets)), budgets):
       # print averageAccuracies[(gamma, b)][0]
        accuracies0.append(averageAccuracies[(gamma, b)])
         
    
    plt.plot(budgets, accuracies0, color=c, linestyle='-', 
             label=p, linewidth=5.0)
    
    print accuracies0

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
ax.legend(handles, labels, bbox_to_anchor=(1, 1), 
          loc=2, borderaxespad=0.0)


plt.show()
plt.clf()

