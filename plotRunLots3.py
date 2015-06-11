#Make worker accuracy bar chart
import matplotlib.pyplot as plt 
import numpy as np
import cPickle as pickle

#budget = 2000
#budgetInterval = 50
#budgets = range(budgetInterval, budget+budgetInterval, budgetInterval)
b = 500
workerAccuracies = [3.3, 1.7, 1.0, 0.5, 0.15]


averageAccuracies = pickle.load(
    open("results/runlotsAccuracies16", 
         "rb"))

#averageErrors = pickle.load(
#    open("results/runlotsErrorst150,625,500,500", 
#         "rb"))


print averageAccuracies

ind = np.arange(5)

ratios1 = []
ratios2 = []
ratios3 = []

fig = plt.figure()   
ax = fig.add_subplot(111)
for (i, gamma) in zip(range(len(workerAccuracies)), workerAccuracies):

 #fig, ax = plt.subplots()
    #ax.set_xlabel('Budget')
    #ax.set_ylabel('Accuracy')
    #ax.set_ylabel('Relabeling Accuracy / Unilabling Accuracy')
    


    ratios1.append(averageAccuracies[(gamma ,b)][1] /
                   averageAccuracies[(gamma, b)][0])
    ratios2.append(averageAccuracies[(gamma ,b)][2] /
                   averageAccuracies[(gamma, b)][0])
    ratios3.append(averageAccuracies[(gamma ,b)][3] /
                   averageAccuracies[(gamma, b)][0])

p1 = plt.bar(ind * 3, ratios1, color='r')
p2 = plt.bar(ind * 3 + 1, ratios2,color='y')
p3 = plt.bar(ind * 3 + 2, ratios3, color='g')

ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position(('data', 1.0))
ax.spines['top'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)

plt.ylabel('Relabeling Accuracy / Unilabeling Accuracy')
#plt.title('')
plt.xticks(ind*3+0.8/2., ('55', '65', '75', '85', '95') )
#plt.yticks(np.arange(0.9,1.1,0.1))
plt.ylim(0.9,1.1)
plt.legend( (p1[0], p2[0], p3[0]), ('2/3', '3/5', '4/7') )

plt.show()
