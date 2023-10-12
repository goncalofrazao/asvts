__author__ = "Gonçalo Frazão, Gustavo Vítor, and Tiago Costa"
__credits__ = ["Gonçalo Frazão", "Gustavo Vítor", "Tiago Costa"]
__license__ = "DIESEL-Net License"
__version__ = "0.0.1"
__maintainer__ = "Gonçalo Frazão"
__email__ = "goncalobfrazao@tecnico.ulisboa.pt"
__maintainer__ = "Gustavo Vítor"
__email__ = "gustavovitor@tecnico.ulisboa.pt"
__maintainer__ = "Tiago Costa"
__email__ = "tiagoncosta@tecnico.ulisboa.pt"
__status__ = "Assignment 1: Finished"

"""
This module contains a class that represents a fleet problem and provides methods 
for loading a problem instance from a file and computing the cost of a solution.

Example usage:
    from solution import FleetProblem

    # Load a problem instance from a file
    with open('problem_instance.txt', 'r') as f:
        fp = FleetProblem(fh=f)

    # Compute the cost of a solution
    sol = [('Dropoff', 0, 0, 60.0), ('Pickup', 0, 0, 20.0)]
    cost = fp.cost(sol)

Attributes:
    P (int): The number of points in the problem instance
    R (int): The number of requests in the problem instance
    V (int): The number of vehicles in the problem instance
    matrix (list): A matrix representing the distances between points
    requests (list): A list of requests, each represented as a list of attributes
    vehicles (list): A list of integers representing the capacities of the vehicles

Todo:
    * Assignment #3

"""

import search
import utils

class Heap(utils.PriorityQueue):
    def __init__(self, order='min', f=lambda x: x):
        self.heap = []
        self.position = {}
        self.free = 0
        if order == 'min':
            self.f = f
        elif order == 'max':  # now item with max f(x)
            self.f = lambda x: -f(x)  # will be popped first
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, key):
        self.heap.append(key)
        self.position[key] = self.free
        self.free += 1
        self._fix_up(self.free - 1)

    def pop(self):
        if self.free == 1:
            self.free = 0
            key = self.heap.pop()
            self.position.pop(key)
            return key
        if self.free > 1:
            i, j = 0, self.free - 1
            self.position[self.heap[i]], self.position[self.heap[j]] = j, i
            self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
            key = self.heap.pop()
            self.position.pop(key)
            self.free -= 1
            self._fix_down(0)
            return key
        else:
            raise Exception('Trying to pop from empty heap.')

    def __len__(self):
        return self.free
    
    def __contains__(self, key):
        return key in self.position
    
    def __getitem__(self, key):
        return self.position[key]
    
    def __delitem__(self, key):
        i, j = self.position[key], self.free - 1
        self.position[self.heap[i]], self.position[self.heap[j]] = j, i
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

        key = self.heap.pop()
        self.position.pop(key)
        self.free -= 1

        self._fix_down(i)

    def _fix_up(self, idx):
        while idx > 0 and self.f(self.heap[idx]) < self.f(self.heap[(idx - 1) // 2]):
            self.position[self.heap[idx]] = (idx - 1) // 2
            self.position[self.heap[(idx - 1) // 2]] = idx
            self.heap[idx], self.heap[(idx - 1) // 2] = self.heap[(idx - 1) // 2], self.heap[idx]
            idx = (idx - 1) // 2

    def _fix_down(self, idx):
        while idx * 2 + 1 < self.free:
            child = 2 * idx + 1
            if child + 1 < self.free and self.f(self.heap[child + 1]) < self.f(self.heap[child]):
                child += 1
            if self.f(self.heap[child]) < self.f(self.heap[idx]):
                self.position[self.heap[idx]], self.position[self.heap[child]] = child, idx
                self.heap[idx], self.heap[child] = self.heap[child], self.heap[idx]
            else:
                break
            idx = child

search.PriorityQueue = Heap

class State:
    def __init__(self, requests):
        self.requests = requests

    def __lt__(self, other):
        return True
    
    def __eq__(self, other):
        return self.requests == other.requests
    
    def __hash__(self):
        return hash(tuple(self.requests))

class FleetProblem(search.Problem):
    
    def __init__(self, fh=None):
        """Constructor method that initializes the attributes of the FleetProblem 
        object and loads a problem instance from file object fh if it is provided.

        Args:
            fh (file object): File object containing the problem instance.

        """
        self.P = 0
        self.R = 0
        self.V = 0
        self.matrix = []
        self.requests = []
        self.vehicles = []
        self.initial = None
        if fh:
            self.load(fh)

    def load(self, fh):
        """Loads a problem from file object fh.

        Args:
            fh (file object): File object containing the problem instance.

        Side effects:
            Initializes the attributes of the FleetProblem object.

        """
        self.__init__()
        p = 0
        r = 0
        v = 0
        for line in fh:
            # Parse the input file line by line
            if p > 0:
                # Read p lines with matrix distances
                self.matrix[self.P - p - 1] = [0] * (self.P - p) + [float(x) for x in line.split()]
                p -= 1
            elif r > 0:
                # Read r lines of requests
                self.requests.append([float(line.split()[0])] + [int(x) for x in line.split()[1:]])
                r -= 1
            elif v > 0:
                # Read v lines of vehicle capacities
                self.vehicles.append(int(line))
                v -= 1
            elif line.startswith('P'):
                # If the line starts with 'P', read the number of points
                self.P = int(line[2:])
                self.matrix = [[0] * self.P] * self.P
                p = self.P - 1
            elif line.startswith('R'):
                # If the line starts with 'R', read the number of requests
                self.R = int(line[2:])
                r = self.R
            elif line.startswith('V'):
                # If the line starts with 'V', read the number of vehicles
                self.V = int(line[2:])
                v = self.V
        
        self.matrix_triangulation()
        self.initial = State([(0,-1,) for i in range(len(self.requests))])

    def matrix_triangulation(self):
        """Fills in the lower triangle of the distance 
        matrix with the values from the upper triangle.

        Side effects:
            Modifies the self.matrix attribute.

        """
        for i in range(self.P):
            for j in range(i + 1, self.P):
                self.matrix[j][i] = self.matrix[i][j]
    
    def get_action_time(self, action):
        """Returns the dropoff time of an action.

        Args:
            action (tuple): A tuple representing an action.

        Returns:
            float: The dropoff time of the action.

        """
        return action[3]
    
    def is_dropoff(self, action):
        """Returns True if an action is a dropoff action, False otherwise.

        Args:
            action (tuple): A tuple representing an action.

        Returns:
            bool: True if the action is a dropoff action, False otherwise.

        """
        return action[0] == 'Dropoff'
        
    def get_request_time(self, action):
        """Returns the request time of an action.

        Args:
            action (tuple): A tuple representing an action.

        Returns:
            float: The request time of the action.

        """
        return self.requests[action[2]][0]
    
    def get_trip_time(self, action):
        """Returns the trip time of an action.

        Args:
            action (tuple): A tuple representing an action.

        Returns:
            float: The trip time of the action.

        """
        request = self.requests[action[2]]
        return self.matrix[request[1]][request[2]]
    
    def cost(self, sol):
        """Computes the cost of solution sol.

        Args:
            sol (list): A list of actions representing a solution.

        Returns:
            float: The cost of the solution.

        """
        cost = 0
        for action in sol:
            if self.is_dropoff(action):
                cost += self.get_action_time(action) - self.get_request_time(action) - self.get_trip_time(action)
        
        return cost
    
    def result(self, state, action):
        car = action[1]
        request = action[2]
        time = action[3]
        requests = [i for i in state.requests]
        if requests[request][0] == 0:
            requests[request] = (1, time, car)
        elif requests[request][0] == 1:
            requests[request] = (2, time, car)
        return State(requests)
    
    def actions(self, state):
        actions = []
        picks = []
        drops = []
        pos = [0] * self.V
        time = [0] * self.V
        occupation = [0] * self.V
        for i, r in enumerate(state.requests):
            s = r[0]
            if s == 0:
                picks.append(i)
            elif s == 1:
                t = r[1]
                car = r[2]
                drops.append((i, car))
                occupation[car] += self.requests[i][3]
                if t > time[car]:
                    time[car] = t
                    pos[car] = self.requests[i][1]
            else:
                t = r[1]
                car = r[2]
                if t > time[car]:
                    time[car] = t
                    pos[car] = self.requests[i][2]

        for i in picks:
            for v in range(self.V):
                if occupation[v] + self.requests[i][3] <= self.vehicles[v]:
                    t = time[v] + self.matrix[pos[v]][self.requests[i][1]]
                    actions.append(('Pickup', v, i, t if t >= self.requests[i][0] else self.requests[i][0]))

        for i, v in drops:
            t = time[v] + self.matrix[pos[v]][self.requests[i][2]]
            actions.append(('Dropoff', v, i, t))
        
        return actions
    
    def goal_test(self, state):
        return all([r[0] == 2 for r in state.requests])

    def path_cost(self, c, state1, action, state2):
        request = action[2]
        if not self.is_dropoff(action):
            return c + self.get_action_time(action) - self.get_request_time(action)
        else:
            return c + self.get_action_time(action) - state1.requests[request][1] - self.get_trip_time(action)

    def solve(self):
        return search.uniform_cost_search(self, display=True).solution()
