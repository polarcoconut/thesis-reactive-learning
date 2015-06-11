import matplotlib.pyplot as plt 
import numpy as np
import cPickle as pickle

budget = 2000
budgetInterval = 50
budgets = range(budgetInterval, budget+budgetInterval, budgetInterval)
workerAccuracies = [3.3, 1.7, 1.0, 0.5, 0.15]


bestStrategies = pickle.load(open(
        "results/runlotsStrategiesHR50,2500,2000,50",
        "rb"))

fig, ax = plt.subplots()
ax.matshow(bestStrategies, cmap = plt.cm.gray_r)
#ax.set_xticklabels(budgets)
plt.xticks(range(len(budgets)), budgets)
ax.xaxis.set_label_position('top')
ax.set_xlabel("Budget")
#ax.set_yticklabels([55, 65, 75, 85, 95])
plt.yticks(range(len(workerAccuracies)), [55,65,75,85,95])
ax.set_ylabel("Worker Accuracy")
plt.show()
