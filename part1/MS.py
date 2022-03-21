from queue import PriorityQueue

import h5py
import numpy as np

"""
Node class representing a state in the grid and storing information for the A* algorithm
"""


class Node:

    def __init__(self, point):
        # List [x,y]
        self.point = point
        # Instance of class Node
        self.parent = None
        # Heuristic function value of this node
        self.H = 0
        # Cost function value of this node, from the start
        self.G = 0

    def __repr__(self):
        return repr(self.point)

    def move_cost(self, other):
        return 1

    def __gt__(self,other):
        return self.H+self.G > other.H+other.G

    def __eq__(self, other):
        return self.point[0] == other.point[0] and self.point[1] == other.point[1]


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
        # Load obsevations
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
        # Part of the path to take into account. For instance, if this value is set to 0.25, the position at 25% of the path with be used to compute scores
        percentagesOfPath = [0.9]
        for i in range(len(obs)):
            o = obs[i]
            for p in percentagesOfPath:
                # Extract the position at the given percentage of the path
                pos = Node(o[int(p * len(o))])
                X.append(pos)
                Y.append(labels[i])
        self.observations = X
        self.labels = Y

        self.size = 64
        # Build the map grid
        f = h5py.File('maps/' + mapName + ".h5", 'r')
        mapInfo = f['map'][:]
        filterSize = (int)(mapInfo.shape[0] / self.size)
        mapNodes = [[None for j in range(self.size)] for i in range(self.size)]
        # Transform to map of nodes
        for i in range(self.size):
            for j in range(self.size):
                # Count walls in region, to do a sort of max padding
                mapInfoCrop = mapInfo[j * filterSize:(j + 1) * filterSize, i * filterSize:(i + 1) * filterSize]
                nbWalls = np.count_nonzero(mapInfoCrop)
                # If there are less walls than empty tiles

                if nbWalls < 5 * (filterSize * filterSize) / 6:
                    node = Node((i, j))
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
        l = len(self.grid) - 1
        for x, y in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx = node.point[0] + x
            ny = node.point[1] + y
            if nx > l or nx < 0:
                continue
            if ny > l or ny < 0:
                continue
            if self.grid[nx][ny] is None:
                continue
            children.append(self.grid[nx][ny])
        return children

    """
    Parameters :
        2 nodes that are not None
    Returns :
        A heuristic function that can be used to estimate the cost of the optimal paths between start and goal
    """

    def heuristic(self, start, goal):
        abs1 = abs(goal.point[1] - start.point[1])
        abs0 = abs(goal.point[0] - start.point[0])
        return abs1 + abs0

    """
    Parameters :
        start, the start node
        goal, the goal node
    Returns :
        The optimal list of nodes from the start to the goal (each cost is 1, so the optimal path can be assumed to be the shortest one)
        So to get the cost of the total path, just use the len() function on the result
        If no path is found, the function returns []
    """

    @staticmethod
    def list_node(node):
        cnode = node
        r = []
        while cnode is not None:
            r.append(cnode)
            cnode = cnode.parent
        r.reverse()
        return r

    def planner(self, start: Node, goal: Node):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i][j] is not None:
                    self.grid[i][j].parent = None

        to_explore = PriorityQueue()
        to_explore.put(start)
        while not to_explore.empty():
            node = to_explore.get()
            for cnode in self.children(node):
                cost = node.G + node.move_cost(cnode)
                if cnode.parent is not None:
                    if cnode.G >= cost:
                        cnode.G = cost
                        cnode.parent = node
                else:
                    cnode.H = self.heuristic(cnode, goal)
                    cnode.G = cost
                    cnode.parent = node
                    if cnode == goal:
                        path = World.list_node(cnode)
                        return path
                    to_explore.put(cnode)

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
        proba = np.zeros(len(self.goals))
        pos.point = list(pos.point)
        self.start.point = list(self.start.point)
        for i, g in enumerate(self.goals):
            g.point = list(g.point)
            plan_s_g = self.planner(self.start, g)
            plan_p_g = self.planner(pos, g)
            diff_coss_p_s = len(plan_p_g) - len(plan_s_g)
            proba[i] = np.exp(-diff_coss_p_s) / (1 + np.exp(-diff_coss_p_s))
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
        The accuracy, which is a percentage from 0 to 100, representing the ratio of correct predictions. The formula is given in the pdf file.
        /!\ Be careful and read the pdf instrutions regarding probability ties.
    Tips :
        You should use : self.goals
    """

    def accuracy(self, probabilities, true_goals):
        cor = 0
        for i in range(len(probabilities)):
            cp = probabilities[i]
            max_value = max(cp)
            for j in range(len(cp)):
                if max_value == cp[j]:
                    if true_goals[i] == self.goals[j]:
                        cor += 1
                        break
        return cor


"""This part is the main code to run everything. Do not touch it."""

# The maps to use during computation.
maps = ["BigGameHunters", "CrashSites", "Desolation", "Brushfire", "Caldera", "LakeShore", "Enigma", "WinterConquest"]
# You algorithm should ouput these accuracies
ref_accuracy = [100, 90, 100, 100, 99, 99, 100, 100]
