from MDP import *
from math import sqrt, log
from random import sample

class UCT:
    
    def __init__(self, MDP, c, nos):
        self.N = {} #keeps track of how many times each state was visited
        self.Q = {} # Q-values
        self.MDP = MDP
        self.c = c 
        self.maxDepth = 4
        self.numberOfSimulations = nos 

    def getGreedyAction(self, state):
        bestActions = []
        bestValue = -12938102938102
        for action in self.MDP.getActions(state):
            value =  self.MDP.getHeuristic(state, action)
            if value == bestValue:
                bestActions.append(action)
            elif value > bestValue:
                bestActions = [action]
                bestValue = value
        #print "HERE"
        #print bestActions
        #print bestValue
        return sample(bestActions, 1)[0]
        

    def getBestAction(self, state):
        bestActions = []
        bestValue = -12039810293801 
        for action in self.MDP.getActions(state):
            if self.N[(tuple(state), action)] == 0:
                value = 102430102301231
            else:
                value = (self.Q[(tuple(state), action)]
                         + self.c * sqrt(log(self.N[tuple(state)]) / 
                                    self.N[(tuple(state), action)]))
                #print "Original Value"
                #print self.Q[(tuple(state), action)]
                #print "Extra value"
                #print self.c * sqrt(log(self.N[tuple(state)]) / 
                #                    self.N[(tuple(state), action)])
                #print value
            if value == bestValue:
                bestActions.append(action)
            elif value > bestValue:
                bestActions = [action]
                bestValue = value
        #print "Best actions"
        #print bestActions
        return sample(bestActions, 1)[0]
        
    def rollout(self, state, depth):
        if self.MDP.isTerminal(state):
            return 0
            #return self.MDP.getReward(state)
        if depth > self.maxDepth:
            state[-1] = 0
            return self.MDP.getReward(state)

        action = self.getGreedyAction(state)

        transitions = self.MDP.T(state, action)
        r = random()
        lastp = 0.0
        nextState = None
        for (p, s) in transitions:
            if r >= lastp and r < lastp + p:
                nextState = s
                break
            lastp += p
        if nextState == None:
            print transitions
        reward = (self.MDP.getReward(nextState) + 
                  self.rollout(nextState, depth+1))

        return reward

    def simulate(self, state, depth):
        if self.MDP.isTerminal(state):
            #print "THERE!"
            #print self.MDP.getReward(state)
            return 0
            #return self.MDP.getReward(state)
        if depth > self.maxDepth:
            state[-1] = 0
            #print "HERE"
            #print self.MDP.getReward(state)
            return self.MDP.getReward(state)

        #print state
        if not tuple(state) in self.N:
            self.N[tuple(state)] = 0
            for action in self.MDP.getActions(state):
                self.N[(tuple(state),action)] = 0 
                self.Q[(tuple(state), action)] = 0
            return self.rollout(state, depth)

        action = self.getBestAction(state)
        #print action

        transitions = self.MDP.T(state, action)
        r = random()
        lastp = 0.0
        nextState = None
        for (p, s) in transitions:
            if r >= lastp and r < lastp + p:
                nextState = s
                break
            lastp += p
        if nextState == None:
            print transitions
            
        #print "HMM"
        #print depth
        #print self.MDP.getReward(nextState)
        reward = self.MDP.getReward(nextState) + self.simulate(nextState, 
                                                               depth+1)
        

        #print "THE REWARD"
        #print reward
        #print state
        self.N[tuple(state)] += 1
        self.N[(tuple(state), action)] += 1

        self.Q[(tuple(state), action) ] = self.Q[(tuple(state), action)] + (
            (reward - self.Q[(tuple(state), action)]) / 
            self.N[(tuple(state), action)])

        return reward


    def run(self, state):
        #print state
        for i in range(self.numberOfSimulations):
            #print "Sim %d" % i
            self.simulate(state, 0)
        bestActions = []
        bestValue = -12039810293801 
        print self.MDP.getActions(state)
        for action in self.MDP.getActions(state):
            if self.N[(tuple(state), action)] == 0:
                value = 102430102301231
            else:
                value = self.Q[(tuple(state), action)]
            print value
            if value == bestValue:
                bestActions.append(action)
            elif value > bestValue:
                bestActions = [action]
                bestValue = value
        #print bestValue
        print state[-2]
        return sample(bestActions, 1)[0]
