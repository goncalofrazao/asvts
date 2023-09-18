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
    * Assignment #2
    * Assignment #3

"""

import search

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
    
    def matrix_triangulation(self):
        """Fills in the lower triangle of the distance 
        matrix with the values from the upper triangle.

        Side effects:
            Modifies the self.matrix attribute.

        """
        for i in range(self.P):
            for j in range(i + 1, self.P):
                self.matrix[j][i] = self.matrix[i][j]
    
    def get_dropoff_time(self, action):
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
                cost += self.get_dropoff_time(action) - self.get_request_time(action) - self.get_trip_time(action)
        
        return cost