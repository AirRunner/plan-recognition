import h5py,sys
import numpy as np
from datetime import datetime as time
import random
from copy import deepcopy
import math
#np.set_printoptions(threshold=np.nan)

"""
Node class representing a state in the grid and storing information for the A* algorithm
Do not change this class
"""
class Node:

    def __init__(self,point):
        #List [x,y]
        self.point = point
        #Instance of class Node
        self.parent = None
        #Heuristic function value of this node
        self.H = 0
        #Cost function value of this node, from the start
        self.G = 0

    def move_cost(self,other):
        return 1

"""
World class representing the map.
Variables :
    grid is a list of size 64x64. Walkable positions are instances of Node. Obstacles are None.
        For example, grid[1][4] is either None or a Node with a point variable of [1,4]
    observations is a list of positions [x,y] in the grid. Each element [x,y] represent the last observed position of an agent.
    labels are the goals corresponding to the the observations, in the same order.
"""
class World:

    def __init__(self, mapName):
        #Load obsevations
        f = h5py.File('obs/' + mapName + "Obs.h5", 'r')
        obs = [o[:] for o in f["obs"].values()]
        labels = [l[:] for l in f["labels"].values()]
        self.start = Node(f["start"][:])
        self.goals = []
        for g in f["goals"][:]:
            self.goals.append(Node(g))
        f.close()
        X = []
        Y = []
        #Part of the path to take into account. For instance, if this value is set to 0.25, the position at 25% of the path with be used to compute scores
        percentagesOfPath = [0.9]
        for i in range(len(obs)):
            o = obs[i]
            for p in percentagesOfPath:
                #Extract the position at the given percentage of the path
                pos = Node(o[int(p*len(o))])
                X.append(pos)
                Y.append(labels[i])
        self.observations = X
        self.labels = Y

        self.size = 64
        #Biuld the map grid
        f = h5py.File('maps/' + mapName + ".h5", 'r')
        mapInfo = f['map'][:]
        filterSize = (int)(mapInfo.shape[0]/self.size)
        mapNodes = [[None for j in range(self.size)] for i in range(self.size)]
        #Transform to map of nodes
        for i in range(self.size):
            for j in range(self.size):
                #Count walls in region, to do a sort of max padding
                mapInfoCrop = mapInfo[j*filterSize:(j+1)*filterSize,i*filterSize:(i+1)*filterSize]
                nbWalls = np.count_nonzero(mapInfoCrop)
                #If there are less walls than empty tiles
                node = Node((i,j))
                if nbWalls<5*(filterSize*filterSize)/6:
                    mapNodes[i][j] = node
        f.close()
        self.grid = mapNodes

    """
    Parameters :
        node, an instance of the Node class
    Returns :
        The children of node in the grid (ie the next positions at the left, top, right and bottom after taking a step)
        They must be legal positions (ie not walls or positions outside the map)
    """
    def children(self, node):
        children = []
        #ADD CODE HERE
        return children

    """
    Parameters :
        2 nodes that are not None
    Returns :
        A heuristic function that can be used to estimate the cost of the optimal paths between start and goal
    """
    def heuristic(self, start, goal):
        #ADD CODE HERE
        return 0

    """
    Parameters :
        start, the start node
        goal, the goal node
    Returns :
        The optimal list of nodes from the start to the goal (each cost is 1, so the optimal path can be assumed to be the shortest one)
        So to get the cost of the total path, just use the len() function on the result
        If no path is found, the function returns []
    """
    def planner(self, start, goal):
        #ADD CODE HERE
        print("Planner warning : no path found. Returning []")
        return []

    """
    Parameters :
        pos, the agent position [x,y] in the grid
    Returns :
        The posterior probability distribution as defined with the algorithm in the pdf file. It is important to return a list that is in the same order as self.goals
    Tips :
        You should use : self.start, self.goals and the functions you filled
    """
    def predictMastersSardina(self, pos):
        #ADD CODE HERE
        proba = [0]*len(self.goals)
        return proba


    """
    Parameters :
        probabilities, a list of posterior probability distributions (returned by predictMastersSardina). It is thus a list of list.
        true_goals, a list of the true goals sought by the agent, in the same order as probabilities

        For instance, if :
        probabilities[0] = [0.5, 0.2, 0.2, 0.1, 0.0]
        true_goals[0] = [50,62]
        self.goals = [[50,62],[100,20],[10,25],[30,0],[82,67]]
        then it is a correct prediction.
    Returns :
        The accuracy, which is a percentage from 0 to 100, respresenting the ratio of correct predictions. The formula is given in the pdf file.
        /!\ Be careful and read the pdf instrutions regarding probability ties.
    Tips :
        You should use : self.goals
    """
    def accuracy(self, probabilities, true_goals):
        #ADD CODE HERE
        return 0


"""This part is the main code to run everything. Do not touch it."""

#The maps to use during computation.
maps = ["BigGameHunters", "CrashSites", "Desolation", "Brushfire", "Caldera", "LakeShore", "Enigma", "WinterConquest"]
#You algorithm should ouput these accuracies
ref_accuracy = [100, 90, 100, 100, 99, 99, 100, 100]

