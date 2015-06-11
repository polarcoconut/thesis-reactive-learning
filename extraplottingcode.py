
#Plots the average number of examples labeled
def makeStat4Plot(f, name, style, color, interval):

    numPoints = budget - startingPoint

    averageNumTimesRelabeled = []
    allLineData = []


    makeXAxis = False
    numSamples = 0.0
    for line in f:
        numSamples += 1.0
        line = line.split('\t')
        #print len(line)
        tempAllLineData = []
        tempAverageLine = 0
        for i in range(len(line) - 1):
            if not makeXAxis:
                x = []
                for j in range(len(line) - 1):
                    x.append(startingPoint + j*interval)

                averageNumTimesRelabeled = [0.0 for k in range(len(x))]
                allLineData = [[] for k in range(len(x))]
                makeXAxis = True
            stats  = line[i].split(',')
            allLineData[i].append(x[i] - float(stats[1]) + float(stats[0]))
            averageNumTimesRelabeled[i] += (x[i] - 
                                            float(stats[1]) + float(stats[0])) 

    for i in range(len(averageNumTimesRelabeled)):
        averageNumTimesRelabeled[i] /= numSamples


    stdLine = [1.96 * np.std(y) / sqrt(numSamples) for y in allLineData]
    print stdLine
    print x
    print averageNumTimesRelabeled
    print len(stdLine)
    print len(x)
    print len(averageNumTimesRelabeled)
    ax.errorbar(x, averageNumTimesRelabeled, yerr=stdLine, linewidth=1, 
                label=name, linestyle=style, color=color)
    handles, labels = ax.get_legend_handles_labels()

#Plots the average number of times relabeled
def makeStat2Plot(f, name, style, color, interval):

    numPoints = budget - startingPoint

    averageNumTimesRelabeled = []
    allLineData = []


    makeXAxis = False
    numSamples = 0.0
    for line in f:
        numSamples += 1.0
        line = line.split('\t')
        #print len(line)
        tempAllLineData = []
        tempAverageLine = 0
        for i in range(len(line) - 1):
            if not makeXAxis:
                x = []
                for j in range(len(line) - 1):
                    x.append(startingPoint + j*interval)

                averageNumTimesRelabeled = [0.0 for k in range(len(x))]
                allLineData = [[] for k in range(len(x))]
                makeXAxis = True
            stats  = line[i].split(',')
            allLineData[i].append(float(stats[1]))
            averageNumTimesRelabeled[i] += float(stats[1])

    for i in range(len(averageNumTimesRelabeled)):
        averageNumTimesRelabeled[i] /= numSamples


    stdLine = [1.96 * np.std(y) / sqrt(numSamples) for y in allLineData]
    print stdLine
    print x
    print averageNumTimesRelabeled
    print len(stdLine)
    print len(x)
    print len(averageNumTimesRelabeled)
    ax.errorbar(x, averageNumTimesRelabeled, yerr=stdLine, linewidth=1, 
                label=name, linestyle=style, color=color)
    handles, labels = ax.get_legend_handles_labels()
