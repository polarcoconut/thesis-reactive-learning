#This file is just for testing random shit.
from utils import *
#import numpy as np
#from scipy.optimize import curve_fit, leastsq, fsolve, newton, brentq
#from sympy.solvers import solve
#from sympy import Symbol
#from sympy import *
#import mpmath as mp
#import matplotlib.pyplot as plt
#import matplotlib

print "QUALITY"
print computeQuality(0.75, 1)
print computeQuality(0.75, 2)
print computeQuality(0.75, 3)
raise Exception 
print computeQuality(0.6, 40)
print computeQuality(0.6, 50)
print computeQuality(0.95, 3)

def func2(x):
    return x ** 2 - 1

def testPAC(e, d, vc, noise):
    print e * log(2.0 / d, 2.0)
    print vc
    print e
    print  8.0 * vc / e * log(8.0 * vc / e, 2)

    N = max((4.0 / e * log(2.0 / d, 2.0), 8.0 * vc / e * log(8.0 * vc / e, 2)))
    
    print N
    term1 = 2.0 / (((e/2.0) ** 2) * ((1.0-2.0 * noise) ** 2))
    term2 = log((2.0 * N) / (2.0 * d / 3.0))
    
    print "TERM1"
    print term1
    print term2
    return term1 * term2

print "TESTING PAC"
print testPAC(0.2, 0.1, 10.0, 0.3)

#print fsolve(func2, 0.1)

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

f = open('pacboundcurves', 'w')

eps = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
#styles = ('r-x', 'b-o', 'g-s', 'y-*', 'k-^')
#styles = ('r-x', 'b-o', 'g-s', 'y-*', 'k-^')
#styles = ('r--', 'b-.', 'g-')
#styles = ('r--', 'b-.', 'g-', 'y:', 'k_')
colors = ('#fcae91', '#de2d26', '#a50f15')
styles = ('--', '-.', '-')
#ms = (10,25,50,75,100,125,150,175,200,300,400,500,600,700,800,1200)
budget = 1000
ms = range(1, budget+1, 1)
xs = []
#accuracies = (0.55, 0.65,0.75,0.85, 0.95)
accuracies = (0.55, 0.75, 0.95)
relabelingStrats = (1, 3, 5, 7)
#vc = 1.0
vcs = (1.0, 1000.0, 100000.0)
oogc = 1.0
#As OOGC increases, it means you need more labels for the same epsilon. That means if you fix 
tau = 0.1
delta = 0.1
#Low tau means SQ cannot learn with noise - need more labels for the same error
#As tau goes down, with the same number of labels, error goes up. 
#High tau means it can tolerate lots of noise - need fewer noisy labels for the same error. Or, as tau goes up, error goes down.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Number of Training Examples')
ax.set_ylabel('Epsilon, Upper Bound on Error')
#for e in eps:

a = 0.

#for (a,s, color) in zip(accuracies, styles, colors):
for (vc, s, color) in zip(vcs, styles, colors):
#for (b, s) in zip(budgets, styles):
#for (strat, s) in zip(relabelingStrats, styles):
    print "accuracy"
    print a
    print "number of examples to minimize epsilon"
    x0,y0,c,k = computeLogisticCurve(a, budget)

    print (x0, y0, c, k)
    for x in range(1, 10, 1):
        print x
        print computeQuality(a, x)
        print computeQualityCont(x, x0, y0, c, k)

    results = []
    minEpsilon = 1000.0
    bestNumberOfExamples = 0.0
    xs = []
    
    #ms = [strat * i for i in range(1, 1000/strat)]
    for m in ms:
        minNumberOfLabels = int(budget / m)
        
        examplesWithOneMoreLabel = budget - (m * minNumberOfLabels) 
        examplesWithMinLabels = m - examplesWithOneMoreLabel

        noise1 = 1.0 - computeQualityCont(minNumberOfLabels + 1, x0,y0,c,k)
        noise2 = 1.0 - computeQualityCont(minNumberOfLabels, x0, y0, c, k)

        #noise1 = 1.0 - computeQuality(a, minNumberOfLabels + 1)
        #noise2 = 1.0 - computeQuality(a, minNumberOfLabels)

        #if minNumberOfLabels % 2 == 0:
        #    noise1 = 1.0 - computeQuality(a, minNumberOfLabels + 1)
        #    noise2 = (0.5 * (1.0 - computeQuality(a, minNumberOfLabels + 1)) +
        #             0.5 * (1.0 - computeQuality(a, minNumberOfLabels - 1)))
        #else:
        #    noise1 = (0.5 * (1.0 - computeQuality(a, minNumberOfLabels + 2)) +
        #              0.5 * (1.0 - computeQuality(a, minNumberOfLabels)))
        #    noise2 = 1.0 - computeQuality(a, minNumberOfLabels)


        noise = ((examplesWithMinLabels * noise2) + (examplesWithOneMoreLabel * noise1)) / m

        if m == 333:
            print "WHATS THE NOISE"
            print noise

        xs.append((examplesWithMinLabels * minNumberOfLabels + examplesWithOneMoreLabel * (minNumberOfLabels + 1)) / m)

        #noise = 1.0 - computeQuality(a, strat)
        #noise = 1.0 - computeQualityCont(strat, x0, y0, c, k)

        #print computeQualityCont(1, x0, y0, c, k)
        #noise = max(noise1, noise2)
        #print "HERE"
        #print computeQuality(a, minNumberOfLabels)
        #print examplesWithOneMoreLabel
        #print examplesWithMinLabels
        #print minNumberOfLabels
        #print (x0, y0, c, k)
        #print noise1
        #print noise2
        #print noise

        if np.isinf(noise):
            noise = 0.0

    #print e
        #print m
        e = fsolve(calcPAC, 0.1, args = (oogc, m, tau, delta, vc, a, noise, budget))
        #e = fsolve(calcPAC2, 0.1, args=(m, tau, delta, vc, a, noise, budget))
        #Sanity Check
        #if calcPAC(e, 1, m, 0.2, 0.2, vc, a, budget) != 0:
        #    print calcPAC(e, 1, m, 0.2, 0.2, vc, a, budget)
        #    print "HUH"

        if m == 333:
            print "WHATS THE ERROR"
            print e

        results.append(e)
        if e <= minEpsilon:
            minEpsilon = e
            bestNumberOfExamples = m
    print bestNumberOfExamples
    print minEpsilon

    #print xs
    plt.plot(ms, results, s, color=color, label="s: %.2f" % vc, linewidth = 5.0)
    #m1, m2, k = fsolve(calcPAC, (200, 100, 1000000), args=((0.1, 0.2), 0.1, 0.2, 1000, 0.6, budget)) 

#(x0, y0, c, k)
handles, labels = ax.get_legend_handles_labels()
fig.set_facecolor('white')
#ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0)

plt.show()



"""
e,k,m = symbols('e k m')

print solve(((1.0 / ((0.1 ** 2) * (e ** 2) * ((1.0 - (2 * (0.5845 / (1 + mp.e ** (-0.2964 * (((150-m) / m) - 0.000155))) + 0.2077))) ** 2))) * (log(1.0/e) ** 2) * ((1000 + (1000 * log(1.0 / e) * log(log(1.0 / e)))) * (log((1.0 / (0.1 * e * (1.0 - (2.0 * (0.5845 / (1 + mp.e ** (-0.2964 * (((150-m) / m) - 0.00015))) + 0.2077))))) * log(1.0/e))) + log(1.0/0.2))) / k - m, e, m, k, warning=True, dict = True, minimal=True, quick=True, force=True )
"""








"""
gammas = [3.0]

xs = [3,4,5,6]
for g in gammas:
    print "Gamma"
    print g
    for numLabels in xs:
        print numLabels
        acc = 0.5 * (1.0 + (1.0 - 0.5) ** g)
        q = computeQuality(acc, numLabels)
        print q

"""




