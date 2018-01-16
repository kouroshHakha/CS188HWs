# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.QValues = util.Counter()
        self.policy = util.Counter()
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = mdp.getStates()
        for state in states:
            self.QValues[state] = util.Counter()

        prevValues = util.Counter()
        for iter in range(self.iterations):
            for state in states:
                for action in mdp.getPossibleActions(state):
                    nextState = [item[0] for item in mdp.getTransitionStatesAndProbs(state, action)]
                    T = [item[1] for item in mdp.getTransitionStatesAndProbs(state, action)]
                    Qs = []
                    if not len(T)==0:
                        Qs = [T[i] * (mdp.getReward(state, action, nextState[i]) + self.discount * prevValues[nextState[i]]) for i in range(len(T))]
                    self.QValues[state][action] = sum(Qs)
                self.policy[state] = self.QValues[state].argMax()
                self.values[state] = self.QValues[state][self.policy[state]]
            prevValues = self.values.copy()

        # print self.policy


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        nextState = [item[0] for item in self.mdp.getTransitionStatesAndProbs(state, action)]
        T = [item[1] for item in self.mdp.getTransitionStatesAndProbs(state, action)]
        Qs = []
        if not len(T)==0:
            Qs = [T[i] * (self.mdp.getReward(state, action, nextState[i]) + self.discount * self.values[nextState[i]]) for i in range(len(T))]

        return sum(Qs)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        Q = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            Q[action] = self.computeQValueFromValues(state, action)

        return Q.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
