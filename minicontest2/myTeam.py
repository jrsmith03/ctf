# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
import util
from game import Directions
import game
import pickle

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########




import random
import util

def nearestPoint( pos ):
    """
    Finds the nearest grid point to a position (discretizes).
    """
    ( current_row, current_col ) = pos

    grid_row = int( current_row + 0.5 )
    grid_col = int( current_col + 0.5 )
    return ( grid_row, grid_col )
class QLearningAgent(CaptureAgent):
    def __init__(self, index, alpha=0.5, gamma=0.9, epsilon=0.25):
        CaptureAgent.__init__(self, index)
        self.q_table = util.Counter()  # Q-values stored as (state, action): value
        # self.loadQTable()  # Load Q-table at the start of the game

        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon # Greedy strategy

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.loadQTable()  # Load Q-table at the start of the game
        self.start = gameState.getAgentPosition(self.index)

    # def final(self, gameState):
    #     print("Hello? Does this ever get called?")
    #     self.saveQTable()  # Save Q-table at the end of the game
    #     CaptureAgent.final(self, gameState)
    #     self.epsilon = epsilon  # Exploration rate

    def getQValue(self, state, action):
        return self.q_table[(state, action)]

    def updateQValue(self, gameState, state, action, nextState, reward):
      maxQ = max([self.getQValue(nextState, a) for a in gameState.getLegalActions(self.index)], default=0)
      currentQ = self.getQValue(state, action)
      self.q_table[(state, action)] = currentQ + self.alpha * (reward + self.gamma * maxQ - currentQ)
    #   print("Current qval: ", currentQ, " temp dist update: ", currentQ + self.alpha * (reward + self.gamma * maxQ - currentQ))
      # Move saveQTable to the final method instead of calling here
      self.saveQTable()  # Save Q-table at the end of the game

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def chooseAction(self, gameState):
        legalActions = gameState.getLegalActions(self.index)
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            state = self.getStateRepresentation(gameState)
            # Choose the best action given the current Qtable
            action = max(legalActions, key=lambda a: self.getQValue(state, a), default=random.choice(legalActions))
            nextState = self.getSuccessor(gameState, action) # action is a movement vector
            reward = self.getReward(gameState, action) # Calculate the given reward based on the overridden method of Offsensive or Defensive.
            self.updateQValue(gameState, state, action, nextState, reward)
            return action

    def getStateRepresentation(self, gameState):
        """
        Subclasses must implement this method to define the state representation.
        """
        raise NotImplementedError

    def getReward(self, gameState, action):
        """
        Subclasses must implement this method to define the reward function.
        """
        raise NotImplementedError
    
    def saveQTable(self, filename="qtable.pkl"):
        # print("Saving Q-Table")
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def loadQTable(self, filename="qtable.pkl"):
        # print("Loading Q-Table")
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            self.q_table = util.Counter()  # Initialize an empty Q-table if not found
            
# we are pacman
class OffensiveQLearningAgent(QLearningAgent):
    def getStateRepresentation(self, gameState):
        # Use a simplified state for Q-learning (e.g., position, food distance)
        position = gameState.getAgentPosition(self.index)
        foodList = self.getFood(gameState).asList()
        closestFoodDist = min([self.getMazeDistance(position, food) for food in foodList]) if foodList else 0
        return (position, closestFoodDist)

    def getReward(self, gameState, action):
        # Reward for eating food or returning food
        successor = self.getSuccessor(gameState, action)
        currentFood = len(self.getFood(gameState).asList())
        nextFood = len(self.getFood(successor).asList())
        reward = currentFood - nextFood + 10 # +1 for each food eaten
        if not successor.getAgentState(self.index).isPacman:
            reward += 10  # Bonus for returning food safely
        if successor.getAgentState(self.index).getPosition() == self.start:
            reward += 5  # Bonus for returning home
        return reward
    
# we are a spoooky ghost
class DefensiveQLearningAgent(QLearningAgent):
    def getStateRepresentation(self, gameState):
        # Use a simplified state for Q-learning (e.g., position, invaders nearby)
        position = gameState.getAgentPosition(self.index)
        invaders = [a for a in self.getOpponents(gameState) if gameState.getAgentState(a).isPacman]
        closestInvaderDist = min([self.getMazeDistance(position, gameState.getAgentPosition(i)) 
                                  for i in invaders if gameState.getAgentPosition(i) is not None], default=0)
        return (position, closestInvaderDist)

    def getReward(self, gameState, action):
        # Reward for stopping invaders or patrolling
        successor = self.getSuccessor(gameState, action)
        invaders = [a for a in self.getOpponents(successor) if successor.getAgentState(a).isPacman]
        # Incentivize getting closer to Pacman
        reward = 1
        # print(self.getOpponents(successor))
        # closestOpponentDist = min([self.getMazeDistance(successor.getAgentPosition(self.index), self.getOpponents(successor)) for opponent in invaders], default=0) + 1
        if any(successor.getAgentPosition(a) == self.start for a in invaders):
            reward -= 100  # Penalty for allowing invaders
        if len(invaders) > 0:
            reward += 5  # Reward for keeping invaders away

        # reward getting close to blue team agent - to cross over to op's side
        blueFood = self.getFood(successor).asList()
        # print(blueFood)
        closestBlueFoodDist = min([self.getMazeDistance(successor.getAgentPosition(self.index), food) for food in blueFood], default=0) + 1
        reward += 1/closestBlueFoodDist #+ 1/closestOpponentDist

        return reward

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveQLearningAgent', second='DefensiveQLearningAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]