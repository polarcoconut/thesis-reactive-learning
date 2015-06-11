import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
#import matplotlib
from math import sqrt 
import pickle
import pandas as pd

#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size' : 18}
#font = {'size' : 40}
#plt.rc('font', **font)

#sns.set_palette("PuBuGn")

#sns.set_style("whitegrid")
sns.set_context("poster", font_scale=2.3, rc={"lines.linewidth" : 5.5})
rcParams.update({'figure.autolayout': True})
#rcParams.update({'text.usetex': True})
#plt.ioff()

#This is the default palette
#sns.set_palette(['#b3cde3', '#8c96c6', '#8856a7', '#810f7c', '#000000'])

#sns.set_palette(sns.dark_palette("muted purple", input="xkcd", reverse=True))
#sns.set_palette(sns.light_palette("green"))
#sns.set_palette(sns.cubehelix_palette(5, start=2, rot=0, dark=0, light=.85))
#numPoints = 150
budget = 2000
#numPoints = 150
#numSamples = 1000
#numSamples = 500
#interval = 12
startingPoint = 50

#x = np.arange(startingPoint,startingPoint+numPoints,1.0)


#Plots the average number of examples relabeled
def makeStatPlot(files, names, style, color, interval, statOption):

    numPoints = budget - startingPoint

    averageNumExamplesRelabeled = []
    allLineData = []

    makeXAxis = False

    snsData = []
    if statOption == 0:
        yaxis = "Number of Examples Relabeld"
    elif statOption == 1:
        yaxis = "Number of Times Relabeled"
    else:
        yaxis = "Number of Examples Labeled"

    print files
    for (f, name) in zip(files, names):
        numSamples = 0.0
        for line in f:
            numSamples += 1.0

            line = line.split('\t')
            tempAllLineData = []
            tempAverageLine = 0
            snsLineData = []
            for i in range(len(line) - 1):
                if not makeXAxis:
                    x = []
                    for j in range(len(line) - 1):
                        x.append(startingPoint + j*interval)
                        
                    averageNumExamplesRelabeled  = [0.0 for k in range(len(x))]
                    allLineData = [[] for k in range(len(x))]
                    makeXAxis = True
                stats  = line[i].split(',')
                averageNumExamplesRelabeled[i] += float(stats[0])
                allLineData[i].append(float(stats[0]))
                if statOption == 0:
                    snsLineData.append(float(stats[0]))
                elif statOption == 1:
                    snsLineData.append(float(stats[1]))
                else:
                    snsLineData.append(x[i] - 
                                       float(stats[1]) + float(stats[0]))



            snsData.append(
                pd.DataFrame({"condition": [name] * len(x),
                              "Number of Labels" : x,
                              "Sampling Unit" : [numSamples] * len(x),
                              yaxis : snsLineData}))

            for i in range(len(averageNumExamplesRelabeled)):
                averageNumExamplesRelabeled[i] /= numSamples
        
    sns.tsplot(pd.concat(snsData), time="Number of Labels", 
               unit="Sampling Unit",
               condition="condition", value=yaxis)


        #stdLine = [1.96 * np.std(y) / sqrt(numSamples) for y in allLineData]
        #print stdLine
        #print x
        #print averageNumExamplesRelabeled
        #print len(stdLine)
        #print len(x)
        #print len(averageNumExamplesRelabeled)
        #ax.errorbar(x, averageNumExamplesRelabeled, yerr=stdLine, linewidth=1, 
        #            label=name, linestyle=style, color=color)




#Plots the number of features vs the amount of relabeling for a given alpha
#and a given number of total labels
#Slice point is the index of the total labels
def makeStatAggPlot(allFiles, samplingStrategies,
                  features, colors, slicePoint = False):

    numPoints = budget - startingPoint

    #print allFiles
    for (i, samplingStrategy) in zip(range(len(samplingStrategies)), 
                                     samplingStrategies):
        numExamplesRelabeled = []
        numTimesRelabeled = []
        numExamplesRelabeledSTD = []
        numTimesRelabeledSTD = []  

        for f in allFiles[i]:
            #print allFiles[i]
            averageNumExamplesRelabeled = 0.0
            averageNumTimesRelabeled = 0.0
            averageNumExamplesRelabeledData = []
            averageNumTimesRelabeledData = []

            numSamples = 0.0
            for line in f:
                numSamples += 1.0
                line = line.split('\t')
                #print line
                if slicePoint == False:
                    slicePoint = len(line)-2
                stats  = line[slicePoint].split(',')
                #print stats
                averageNumExamplesRelabeledData.append(float(stats[0]))
                averageNumTimesRelabeledData.append(float(stats[1]))

                averageNumExamplesRelabeled += float(stats[0])
                averageNumTimesRelabeled += float(stats[1])

            averageNumTimesRelabeled /= numSamples
            averageNumExamplesRelabeled /= numSamples
            
            numExamplesSTD = (1.96 * np.std(averageNumExamplesRelabeledData) / 
                              sqrt(numSamples) )
            
            numTimesSTD = (1.96 * np.std(averageNumTimesRelabeledData) / 
                           sqrt(numSamples))
            
            numExamplesRelabeled.append(averageNumExamplesRelabeled)
            numTimesRelabeled.append(averageNumTimesRelabeled)
            numExamplesRelabeledSTD.append(numExamplesSTD)
            numTimesRelabeledSTD.append(numTimesSTD)
            
        print numTimesRelabeled
        print numExamplesRelabeled
        ax.errorbar(features, numTimesRelabeled, 
                    yerr=numTimesSTD, linewidth=1, 
                    label="Num Examples Relabeled", linestyle='-',
                    color = colors[0][i])
        
        ax.errorbar(features, numExamplesRelabeled, yerr=numExamplesSTD, 
                    linewidth=1, 
                    label="Num Times Relabeled", linestyle='-',
                    color = colors[1][i])



#def makePlot(f, name, style, x, numPoints):
def makePlot(files, names, style, color, interval):
 
    numPoints = budget - startingPoint
    averageLine = []
    allLineData = []

    snsData = []
    #print files
    for (f, name) in zip(files, names):
        makeXAxis = False
        x = []
        snsConditionData = []
        numSamples = 0.0
        for line in f:
            numSamples += 1
            line = line.split('\t')
            tempAllLineData = []
            tempAverageLine = 0
            snsLineData = []
            for i in range(len(line) - 1):
                if not makeXAxis:
                    x = []
                    for j in range(len(line) - 1):
                        x.append(startingPoint + j*interval)

                    averageLine = [0.0 for k in range(len(x))]
                    allLineData = [[] for k in range(len(x))]
                    makeXAxis = True
                #print i
                #print len(line)
                #if len(line) > 80:
                #    print line
                #print float(line[i])
                allLineData[i].append(float(line[i]))
                snsLineData.append(float(line[i]))
                averageLine[i] += float(line[i])
            snsData.append(
                pd.DataFrame({"condition": [name] * len(x),
                              "Number of Labels" : x,
                              "Sampling Unit" : [numSamples] * len(x),
                              "Test Accuracy" : snsLineData}))
        for i in range(len(averageLine)):
            averageLine[i] /= numSamples
    

    ax = sns.tsplot(pd.concat(snsData), time="Number of Labels", 
               unit="Sampling Unit",
               condition="condition", value="Test Accuracy", ci=95)

    """
    ax.lines[0].set_marker("o")
    ax.lines[0].set_markersize(5)
    ax.lines[1].set_marker("^")
    ax.lines[1].set_markersize(5)
    ax.lines[2].set_marker("s")
    ax.lines[2].set_markersize(5)
    ax.lines[3].set_marker("p")
    ax.lines[3].set_markersize(5)
    ax.lines[4].set_marker("*")
    ax.lines[4].set_markersize(5)
    ax.lines[5].set_marker("d")
    ax.lines[5].set_markersize(5)
    #ax.lines[6].set_marker("1")
    #ax.lines[6].set_markersize(5)
    ax.legend()
    """

    stdLine = [1.96 * np.std(y) / sqrt(numSamples) for y in allLineData]
    print stdLine
    print x
    print averageLine
    print len(stdLine)
    print len(x)
    print len(averageLine)
    #ax.errorbar(x, averageLine, yerr=stdLine, linewidth=1, 
    #            label=name, linestyle=style, color=color)
    #ax.errorbar(x, averageLine, yerr=stdLine, 
    #            label=name)
    #plt.errorbar(x, averageLine, yerr=stdLine, label=name)

    #plt.plot(x, averageLine, label=name)

def makeHistogram2(f, interval=12):
    allLineData = []
    makeXAxis = False
    x = []
    for line in f:
        line = line.split('\t')
        for i in range(len(line) - 1):
            if not makeXAxis:
                x = []
                for j in range(len(line) - 1):
                    x.append(startingPoint + j*interval)
                makeXAxis = True
            allLineData.append(int(float(line[i])))

    plt.hist(allLineData, bins=[i for i in range(11)])


def makeHistogram(f, name, style):
    averageLine = [0.0 for i in range(numPoints)]
    allLineData = [[] for i in range(numPoints)]
    
    for line in f:
        line = line.split('\t')
        for i in range(numPoints):
        #print line[i]
            allLineData[i].append(float(line[i]))
            averageLine[i] += float(line[i])
    
    for i in range(len(averageLine)):
        averageLine[i] /= numSamples

    stdLine = [1.96 * np.std(y) / sqrt(numSamples) for y in allLineData]
    ax.errorbar(x, averageLine, yerr=stdLine, linewidth=3, 
                label=name, fmt=style)

#f = open('outputs/bc9/usX-b15-g30', 'r')
#makePlot(f, 'Validation Re-active MDP', 'b-')

#f = open('outputs/bc9/usD-b15-g30', 'r')
#makePlot(f, 'Direct Re-active MDP', 'b--')

#numPoints = 250
#x = np.arange(startingPoint,startingPoint+numPoints,1.0)


#f = open('outputs/fa1/al2-b150-g30', 'r')
#makePlot(f, 'Only Active learning', 'g--')

#numPoints = 200
#x = range(50,100,1)
#y = range(100, 200, 2)

#x = x + y
#f = open('outputs/fa1/relearn2-b150-g30', 'r')
#makePlot(f, 'Only Relabeling', 'r-.')

"""
f = open('outputs/fd1/al-b150-g30-2000,100,0', 'r')
makePlot(f, 'Only New Examples-100,0', 'k-.')

f = open('outputs/fd1/relearn-b150-g30-2000,100,0', 'r')
makePlot(f, 'Only Relabeling-100,0', 'y-.')

f = open('outputs/fd1/al-b150-g30-2000,200,0', 'r')
makePlot(f, 'Only New Examples-200,0', 'rx')

f = open('outputs/fd1/relearn-b150-g30-2000,200,0', 'r')
makePlot(f, 'Only Relabeling-200,0', 'gx')

f = open('outputs/fd1/al-b150-g30-2000,100,100', 'r')
makePlot(f, 'Only New Examples-100,100', 'bo')

f = open('outputs/fd1/relearn-b150-g30-2000,100,100', 'r')
makePlot(f, 'Only Relabeling-100,100', 'co')
"""

##
# PLOT BAR GRAPHS
##

"""
#features = [10,30,50,70,90]
features = [30]
for feature in features:
    f = open('outputs/g7R/impactPriorPLOPT(7)-f%d-lr-g1.0-1000-250-impactStats' %  feature, 'r')
    makeHistogram2(f)
    plt.show()
    plt.clf()
"""




#hues = ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6']
hues = ['#b3cde3', '#8c96c6', '#8856a7', '#810f7c', '#000000']
#
# PLOT THE STUFF
#




#features = [10,30,50,70,90]
features = [50]
budget = 1000
foldername = 'g7R'

for feature in features:
    files = []
    #files.append(open('outputs/g7/cv15-f%d-lr-g1.0-1000-250' % feature, 'r'))
    #files.append(open('outputs/g7/cvBatch-f%d-lr-g1.0-1000-50' % feature, 'r'))

    
    """
    files.append(
        open('outputs/g7/impactPrior-f%d-lr-g1.0-1000-250' % feature, 'r'))
    files.append(
        open('outputs/g7/impactPriorMax-f%d-lr-g1.0-1000-250' % feature, 'r'))
    files.append(
        open('outputs/g7/impactPriorExpectedMax-f%d-lr-g1.0-1000-250' % feature, 'r'))
    files.append(
        open('outputs/g7/impactPriorEMNeighboruncuncLunc0.1unc0.3unc0.5unc0.7unc0.9-f%d-lr-g1.0-1000-250' % feature, 'r'))
    
    #files.append(open('outputs/g7/unc0.9-f%d-lr-g1.0-1000-250' % feature, 
    #                  'r'))
    #files.append(open('outputs/g7/unc0.3-f%d-lr-g1.0-1000-250' % feature, 
    #                  'r'))
    #files.append(open('outputs/g7/unc-r5-f%d-lr-g1.0-1000-250' % feature, 
    #                  'r'))
    #files.append(open('outputs/g7/unc-f%d-lr-g0.0-1000-250' % feature, 
    #                  'r'))
    #files.append(open('outputs/g7/pass-f%d-dt-g1.0-1000-250' % feature, 
    #                  'r'))

    makePlot(files, ['impactEXP',
                     'impactOPT',
                     'impactPLOPT',
                     'impactPLOPT(7)',
                     r'$US_{X_U}$',
                     r'$US^{0.3}_{X}$',
                     r'$US^{3/5}_{X_U}$',
                     r'$US^*_{X_U}$',
                     ], '-', hues[0], 12)
    
    
    """
    
    
    files.append(
        open('outputs/%s/impactPrior(2)-f%d-lr-g1.0-%d-250' % (foldername, feature,budget), 'r'))
    #files.append(
    #    open('outputs/%s/impactPrior(7)-f%d-lr-g1.0-%d-250' % (foldername, feature,budget), 'r'))
    #files.append(
    #    open('outputs/3g7/impactPriorMediumuncuncLunc0.1unc0.3unc0.5unc0.7unc0.9-f%d-lr-g1.0-1000-250' % feature, 'r'))
    #files.append(
    #    open(
    #        'outputs/g7/impactPriorExpectedMax-f%d-lr-g1.0-1000-250' % feature, 'r'))
    #files.append(
    #    open('outputs/g7/impactPriorEMNeighboruncuncLunc0.1unc0.3unc0.5unc0.7unc0.9-f%d-lr-g1.0-1000-250' % feature, 'r'))
    #files.append(
    #    open(
    #        'outputs/g7/impactPriorExpectedExpectedMax2-f%d-lr-g1.0-1000-250' % feature, 'r'))

    #files.append(
    #    open(
    #        'outputs/%s/impactPriorOPT(2)-f%d-lr-g1.0-%d-250' % 
    #        (foldername, feature,budget), 'r'))
    files.append(
        open(
            'outputs/%s/impactPriorPLOPT(2)-f%d-lr-g1.0-%d-250' % 
            (foldername, feature,budget), 'r'))
    files.append(
        open(
            'outputs/%s/impactPriorPLOPT(7)-f%d-lr-g1.0-%d-250' % 
            (foldername, feature,budget), 'r'))
    #files.append(
    #    open(
    #        'outputs/%s/random(7)-f%d-lr-g1.0-%d-250' % 
    #        (foldername, feature,budget), 'r'))
    #files.append(
    #    open(
    #        'outputs/g7R/impactPriorPL-f%d-lr-g1.0-1000-250' % feature, 'r'))


    #files.append(open('outputs/g7/uncPrior0.9-f%d-lr-g1.0-1000-250' % feature, 
    #                  'r'))
    #files.append(open('outputs/g7/uncPrior0.5-f%d-lr-g1.0-1000-250' % feature, 
    #                  'r'))
    files.append(open('outputs/%s/unc-f%d-lr-g1.0-%d-250' % 
                      (foldername, feature, budget), 
                      'r'))
    #files.append(open('outputs/%s/unc-r3-f%d-lr-g1.0-%d-250' % 
    #                  (foldername, feature, budget), 
    #                  'r'))
    files.append(open('outputs/g7/unc0.5-f%d-lr-g1.0-1000-250' % feature, 
                      'r'))
    #files.append(open('outputs/g7/uncBayes-f%d-lr-g1.0-1000-250' % feature, 
    #                  'r'))
    #files.append(open('outputs/%s/unc-f%d-lr-g0.0-%d-250' % 
    #                  (foldername, feature, budget), 
    #                  'r'))
    #files.append(open('outputs/%s/pass-f%d-lr-g1.0-%d-250' % 
    #                  (foldername, feature, budget), 
    #                  'r'))


    #files.append(open('outputs/g7/unc-r3-f%d-lr-g1.0-1000-250' % feature, 
    #                  'r'))
    makePlot(files, ['impactEXP',
                     'impactPLOPT',
                     'impactPLOPT(7)',
                     r'$US_{X_U}$',
                     r'$US^{0.5}_{X}$',
                     ], '-', hues[0], 12)
    
    
    

    """
    samplingStrategies = [r'$US^{0.1}_{X}$',
                          r'$US^{0.3}_{X}$',
                          r'$US^{0.5}_{X}$',
                          r'$US^{0.7}_{X}$',
                          r'$US^{0.9}_{X}$']
    
    for alpha in ['0.1', '0.3', '0.5', '0.7', '0.9']:
        files.append(open('outputs/g7/unc%s-f%d-lr-g%.1f-1000-250' % (
            alpha, feature, 1.0), 'r'))
    makePlot(files, samplingStrategies, '-', hues[0], 12)
    """

    """
    samplingStrategies = [r'$US_{X_U}$',
                          r'$US^{2/3}_{X_U}$',
                          r'$US^{3/5}_{X_U}$',
                          r'$US^{4/7}_{X_U}$',
                          r'$US^{5/9}_{X_U}$']

    files.append(open('outputs/g7/unc0.9-f%d-lr-g%.1f-1000-250' % (
            feature, 1.0), 'r'))    
    for alpha in ['3','5','7','9']:
        files.append(open('outputs/g7/unc-r%s-f%d-lr-g%.1f-1000-250' % (
            alpha, feature, 1.0), 'r'))
    makePlot(files, samplingStrategies, '-', hues[0], 12)
    """




    plt.ylim([0.5,1.0])
    plt.show()
    plt.clf()



"""
for features in [10,30,50,70,90]:
#for features in [10]:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Number of Labels')
    ax.set_ylabel('Test Accuracy')

    #f = open('outputs/g7/unc-f500-lr-g1.0-2000', 'r')
    #makePlot(f, 'US, no relabeling', '-', 'r', 12)


    #f = open('outputs/log7/cv-f%d-dt-g1.0-500' % features, 'r')
    #makePlot(f, 'CV', '-', hues[0], 12)
    
    
    f = open('outputs/g7/unc0.1-f%d-lr-g1.0-500' % features, 'r')
    makePlot(f, 'Re-US Alpha 0.1', '-', hues[0], 12)
    
    f = open('outputs/g7/unc0.3-f%d-lr-g1.0-500' % features, 'r')
    makePlot(f, 'Re-US Alpha 0.3', '-', hues[1],  12)
    
    f = open('outputs/g7/unc0.5-f%d-lr-g1.0-500' % features, 'r')
    makePlot(f, 'Re-US Alpha 0.5', '-', hues[2], 12)
    
    f = open('outputs/g7/unc0.7-f%d-lr-g1.0-500' % features, 'r')
    makePlot(f, 'Re-US Alpha 0.7', '-', hues[3], 12)
    

    f = open('outputs/g7/unc0.9-f%d-lr-g1.0-500' % features, 'r')
    makePlot(f, 'Re-US Alpha 0.9', '-', hues[4], 12)
    
    #f = open('outputs/g7/pass-f500-lr-g1.0-2000', 'r')
    #3makePlot(f, 'Passive 1 Label/Example', '-', 'm', 12)

    #handles, labels = ax.get_legend_handles_labels()
    
    #ax.legend(handles, labels, bbox_to_anchor=(0.5, 0.5), 
    #          loc=2, borderaxespad=0.0)

    #ax.legend(handles, labels)


    plt.ylim([0.5,1.0])
    plt.show()
    
    plt.clf()
"""


##
#NOW PLOT STATS
##


samplingStrategies = ['0.1','0.3','0.5','0.7','0.9']
#features = [10,30,50,70,90]
#features = [30,50,70,90]
#features = [1558]
#budget = 9
#foldername = 'ad7'
for statOption in [0,1,2]:
    for feature in features:
        files = []

        """
        files.append(
            open('outputs/log7/impactPrior-f%d-dt-g1.0-1000-250-stats' % feature, 'r'))
        files.append(
            open('outputs/log7/impactPriorExpectedMax-f%d-dt-g1.0-1000-250-stats' % feature, 'r'))
        files.append(
            open('outputs/log7/impactPriorEMNeighboruncuncLunc0.1unc0.3unc0.5unc0.7unc0.9-f%d-dt-g1.0-1000-250-stats' % feature, 'r'))
        #files.append(
        #    open('outputs/log7/impactPriorSoft-f%d-dt-g1.0-1000-250-stats' % feature, 'r'))
        files.append(open('outputs/log7/unc0.9-f%d-dt-g1.0-1000-150-stats' % feature, 
                          'r'))
        files.append(open('outputs/log7/unc0.5-f%d-dt-g1.0-1000-150-stats' % feature, 
                          'r'))
        files.append(open('outputs/log7/unc-f%d-dt-g0.0-1000-250-stats' % feature, 
                          'r'))
        files.append(open('outputs/log7/pass-f%d-dt-g1.0-1000-250-stats' % feature, 
                          'r'))

        makeStatPlot(files, ['impact', 
        'impactPsuedoLookaheadMax',
                             'impactEMNeighbor',
                         'uncertainty sampling (0.9)',
                         'uncertainty sampling(0.5)',
                             'uncertainty sampling-perfect',
                             'passive',
                         ], '-', hues[0], 12, statOption)
        
        """
        
        
        #files.append(
        #    open('outputs/g7/cvBatch-f%d-lr-g1.0-1000-20-stats' % feature, 
        #         'r'))
        files.append(
            open('outputs/%s/impactPrior(2)-f%d-lr-g1.0-%d-250-stats' % 
                 (foldername, feature, budget), 
        'r'))
        #files.append(
        #    open('outputs/3g7/impactPriorMediumuncuncLunc0.1unc0.3unc0.5unc0.7unc0.9-f%d-lr-g1.0-1000-250-stats' % feature, 'r'))
        #files.append(
        #    open(
        #        'outputs/g7/impactPriorExpectedMax-f%d-lr-g1.0-1000-250-stats' % feature, 'r'))
        #files.append(
        #    open(
        #        'outputs/g7/impactPriorExpectedExpectedMax2-f%d-lr-g1.0-1000-250-stats' % feature, 'r'))
        #files.append(
        #    open(
        #        'outputs/g7/impactPriorEMNeighboruncuncLunc0.1unc0.3unc0.5unc0.7unc0.9-f%d-lr-g1.0-1000-250-stats' % feature, 'r'))


        files.append(
            open(
                'outputs/%s/impactPriorOPT(2)-f%d-lr-g1.0-%d-250-stats' % 
                (foldername, feature, budget), 'r'))
        files.append(
            open(
                'outputs/%s/impactPriorPLOPT(2)-f%d-lr-g1.0-%d-250-stats' % 
                (foldername, feature, budget), 'r'))
        files.append(
            open(
                'outputs/%s/impactPriorPLOPT(7)-f%d-lr-g1.0-%d-250-stats' % 
                (foldername, feature, budget), 'r'))

        #files.append(
        #    open('outputs/g7/impactPriorMax-f%d-lr-g1.0-1000-250-stats' % feature, 
        #'r'))
        #files.append(
        #    open('outputs/g7/uncPrior0.9-f%d-lr-g1.0-1000-250-stats' % feature, 
#                 'r'))
 #       files.append(
 #           open('outputs/g7/uncPrior0.5-f%d-lr-g1.0-1000-250-stats' % feature, 
  #               'r'))
        files.append(
            open('outputs/%s/unc-f%d-lr-g1.0-%d-250-stats' % 
                 (foldername, feature, budget), 
                 'r'))
        #files.append(
        #    open('outputs/g7/unc0.5-f%d-lr-g1.0-1000-250-stats' % feature, 
        #         'r'))
        #files.append(
        #    open('outputs/g7/uncBayes-f%d-lr-g1.0-1000-250-stats' % feature, 
        #         'r'))
        #files.append(
        #    open('outputs/g7/unc-f%d-lr-g0.0-1000-250-stats' % feature, 
        #         'r'))
        #files.append(
        #    open('outputs/g7/unc-r3-f%d-lr-g1.0-1000-250-stats' % feature, 
        #                  'r'))
        makeStatPlot(files, [
            'impactEXP',
            'impactOPT',
            'impactPLOPT',
            'impactPLOPT(7)',
            'uncertainty sampling'],
                     '-', hues[0], 12, statOption)
        
        """
        for samplingStrategy in samplingStrategies:
            files.append(open(
                'outputs/%s/unc%s-f%d-lr-g%.1f-%d-250-stats' % (
                    foldername, samplingStrategy, feature, 1.0, budget), 'r'))
        makeStatPlot(files, samplingStrategies, '-', hues[0], 12, statOption)
        """

        plt.show()
        
        plt.clf()



"""
for features in [10,30,50,70,90]:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Number of Labels')
    ax.set_ylabel('Number of Examples Relabeled')
    #ax.set_xlabel('Number of Labels')
    #ax.set_ylabel('Number of Times Relabeled')
    #ax.set_xlabel('Number of Labels')
    #ax.set_ylabel('Number of Examples labeled')


    #f = open('outputs/g7/unc-f10-lr-g1.0-stats', 'r')
    #makeStat1Plot(f, 'US, no relabeling', '-', 'g', 12)
    #f = open('outputs/log7/cv-f%d-dt-g1.0-500-stats' % features, 'r')
    #makeStat1Plot(f, 'CV', '-', hues[0], 12)


    
    f = open('outputs/g7/unc0.1-f%d-lr-g1.0-500-stats' % features, 'r')
    makeStat4Plot(f, 'Re-US Alpha 0.1', '-', hues[0], 12)
    
    f = open('outputs/g7/unc0.3-f%d-lr-g1.0-500-stats' % features, 'r')
    makeStat4Plot(f, 'Re-US Alpha 0.3', '-', hues[1], 12)
    
    f = open('outputs/g7/unc0.5-f%d-lr-g1.0-500-stats' % features, 'r')
    makeStat4Plot(f, 'Re-US Alpha 0.5', '-', hues[2], 12)
    
    f = open('outputs/g7/unc0.7-f%d-lr-g1.0-500-stats' % features, 'r')
    makeStat4Plot(f, 'Re-US Alpha 0.7', '-', hues[3], 12)
    
    f = open('outputs/g7/unc0.9-f%d-lr-g1.0-500-stats' % features, 'r')
    makeStat4Plot(f, 'Re-US Alpha 0.9', '-', hues[4], 12)
    
    #f = open('outputs/g7/pass-f10-lr-g1.0-stats', 'r')
    #makeStat1Plot(f, 'Passive 1 Label/Example', '-', 'b', 12)
    
    handles, labels = ax.get_legend_handles_labels()
    
    ax.legend(handles, labels, bbox_to_anchor=(0, 1.0), 
              loc=2, borderaxespad=0.0)
    
    
    plt.show()

    plt.clf()
"""


###
# PLOT AGGREGATED STATS
###
#samplingStrategies = [passive(), uncertaintySampling(), 
#                      uncertaintySamplingAlpha(0.0),
#                      uncertaintySamplingAlpha(0.1),
#                      uncertaintySamplingAlpha(0.5),
#                      uncertaintySamplingAlpha(0.9)]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Number of Features')

samplingStrategies = ['0.1','0.3','0.5','0.7','0.9']
colors = [hues, hues]
#colors = [['#ffeda0', '#feb24c', '#f03b20'],
#          ['#ece7f2', '#a6bddb', '#2b8cbe']]
features = [10,30,50,70,90]

#files = [open('outputs/g7/unc0.1-f10-lr-g1.0-stats', 'r'),
#         open('outputs/g7/unc0.1-f30-lr-g1.0-stats', 'r'),
#         open('outputs/g7/unc0.1-f50-lr-g1.0-stats', 'r'),
#         open('outputs/g7/unc0.1-f70-lr-g1.0-stats', 'r'),
#         open('outputs/g7/unc0.1-f90-lr-g1.0-stats', 'r')]

files = []
for samplingStrategy in samplingStrategies:
    filesTemp = []
    for feature in features:
        #filesTemp.append(open('outputs/log7/unc%s-f%d-dt-g%.1f-500-stats' % (
        #    samplingStrategy, feature, 1.0), 'r'))
        filesTemp.append(open('outputs/log7/cv-f%d-dt-g%.1f-500-stats' % (
            feature, 1.0), 'r'))
    files.append(filesTemp)
    #files.append(open('outputs/g7/%s-f%d-lr-g%.1f' % (
    #samplingStrategy.getName(), numFeatures, gamma), 'r'))
    #statfiles.append(open('outputs/g7/%s-f%d-lr-g%.1f-stats' % (
    #samplingStrategy.getName(), numFeatures, gamma), 'r'))

makeStatAggPlot(files, samplingStrategies, features, colors)


handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels, bbox_to_anchor=(1.0, 0.5), 
          loc=2, borderaxespad=0.0)

plt.show()
plt.clf()

"""
f = open('outputs/fd1/relearn3-b300-g30-d0s998,2000,100,0', 'r')
makePlot(f, '3 Label/Example', 'c-.', 12)

f = open('outputs/fd1/relearn5-b300-g30-d0s998,2000,100,0', 'r')
makePlot(f, '5 Label/Example', 'c--', 15)

f = open('outputs/fd1/relearn7-b300-g30-d0s998,2000,100,0', 'r')
makePlot(f, '7 Label/Example', 'c.-', 14)
"""


#f = open('outputs/fd1/weld-b200-g30-2000,200,0', 'r')
#makePlot(f, '3', 'r-.')

#f = open('outputs/fd1/weld-b200-g30-2000,200,0', 'r')
#makePlot(f, '4', 'b-.')


#f = open('outputs/bc10/pac-b150-g30-125', 'r')
#makePlot(f, '175', 'g-.')

#f = open('outputs/bc10/pac-b150-g30-150', 'r')
#makePlot(f, '200', 'c-.')

#f = open('outputs/s1/zhao2-b50-g30', 'r')
#makePlot(f, 'Relabel 3 times', 'y-.')

#f = open('outputs/bc8/accRelearn3-b50-g30', 'r')
#makePlot(f, 'Only Relabeling Accuracy', 'y-.')

#f = open('outputs/bc8/histRelearn3-b50-g30', 'r')
#makePlot(f, 'Histogram', 'g-.')

