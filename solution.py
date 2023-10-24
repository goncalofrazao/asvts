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
__status__ = "Really good mega boosted admissible heuristic finished"

"""
This module contains a class that represents a fleet problem and provides methods 
for loading a problem instance from a file, computing the cost of a solution, 
auxiliar methods for the search algorithms, and a method for solving the problem.

Example usage:
    from solution import FleetProblem

    # Load a problem instance from a file
    with open('problem_instance.txt', 'r') as f:
        fp = FleetProblem(fh=f)

    # Compute the cost of a solution
    sol = [('Dropoff', 0, 0, 60.0), ('Pickup', 0, 0, 20.0)]
    cost = fp.cost(sol)

    # Solve the problem
    sol = fp.solve()

Attributes:
    P (int): The number of points in the problem instance
    R (int): The number of requests in the problem instance
    V (int): The number of vehicles in the problem instance
    matrix (list): A matrix representing the distances between points
    requests (list): A list of requests, each represented as a list of attributes
    vehicles (list): A list of integers representing the capacities of the vehicles

Todo:
    * Assignment #3: Really good heuristic incoming for informed search mega boosted algorithms
"""

import search
import utils
import itertools

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
        item = (self.f(key), key)
        self.heap.append(item)
        self.position[item] = self.free
        self.free += 1
        self._fix_up(self.free - 1)

    def pop(self):
        if self.free == 1:
            self.free = 0
            key = self.heap.pop()
            self.position.pop(key)
            return key[1]
        if self.free > 1:
            i, j = 0, self.free - 1
            self.position[self.heap[i]], self.position[self.heap[j]] = j, i
            self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
            key = self.heap.pop()
            self.position.pop(key)
            self.free -= 1
            self._fix_down(0)
            return key[1]
        else:
            raise Exception('Trying to pop from empty heap.')

    def __len__(self):
        return self.free
    
    def __contains__(self, key):
        return (self.f(key), key) in self.position
    
    def __getitem__(self, key):
        return self.f(key)
    
    def __delitem__(self, key):
        item = (self.f(key), key)
        i, j = self.position[item], self.free - 1
        self.position[self.heap[i]], self.position[self.heap[j]] = j, i
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

        item = self.heap.pop()
        self.position.pop(item)
        self.free -= 1

        self._fix_down(i)

    def _fix_up(self, idx):
        while idx > 0 and self.heap[idx] < self.heap[(idx - 1) // 2]:
            self.position[self.heap[idx]] = (idx - 1) // 2
            self.position[self.heap[(idx - 1) // 2]] = idx
            self.heap[idx], self.heap[(idx - 1) // 2] = self.heap[(idx - 1) // 2], self.heap[idx]
            idx = (idx - 1) // 2

    def _fix_down(self, idx):
        while idx * 2 + 1 < self.free:
            child = 2 * idx + 1
            if child + 1 < self.free and self.heap[child + 1] < self.heap[child]:
                child += 1
            if self.heap[child] < self.heap[idx]:
                self.position[self.heap[idx]], self.position[self.heap[child]] = child, idx
                self.heap[idx], self.heap[child] = self.heap[child], self.heap[idx]
            else:
                break
            idx = child

# search.PriorityQueue = Heap

class State(tuple):
    def __lt__(self, other):
        return sum([x[0] for x in self]) > sum([x[0] for x in other])

class FleetProblem(search.Problem):
    
    def __init__(self, fh=None):
        """Constructor method that initializes the attributes of the FleetProblem 
        object and loads a problem instance from file object fh if it is provided.
    
        Usage:
            If there is a file object, this must be opened in read mode and the file must be
            formatted as follows without a specific order of the sections:
                P <number of points>
                <matrix of distances>
                R <number of requests>
                <requests>
                V <number of vehicles>
                <vehicle capacities>
                Where:
                    <matrix of distances> is a upper triangular matrix of floats or integers
                    <requests> are represented by a line with the following format:
                        <request time> <pickup point> <dropoff point> <number of passengers>
                    <vehicle capacities> are represented by a line with a single integer

        Args:
            fh (file object): File object containing the problem instance.

        """
        self.P = 0
        self.R = 0
        self.V = 0
        # matrix is a square symetric matrix
        # where matrix[i][j] is the distance between points i and j
        self.matrix = []
        # request format: (<request time>, <pickup point>, <dropoff point>, <number of passengers>)
        self.requests = []
        # each vehicle is represented by its capacity
        self.vehicles = []
        # initial state is a tuple of tuples
        # where each tuple represents a request state
        # format: (<request state>) if request have not been picked up
        #         (<request state>, <time>, <vehicle>) any other case
        self.initial = ()
        if fh:
            self.load(fh)

    def load(self, fh):
        """Loads a problem from file object fh.

        Args:
            fh (file object): File object containing the problem instance.
        
        Usage:
            The file object must be opened in read mode and the file must be
            formatted as follows without a specific order of the sections:
                P <number of points>
                <matrix of distances>
                R <number of requests>
                <requests>
                V <number of vehicles>
                <vehicle capacities>
                Where:
                    <matrix of distances> is a upper triangular matrix of floats or integers
                    <requests> are represented by a line with the following format:
                        <request time> <pickup point> <dropoff point> <number of passengers>
                    <vehicle capacities> are represented by a line with a single integer
                
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
        self.initial = State((0,) for _ in range(len(self.requests)))
        
        indexed_vehicles = list(enumerate(self.vehicles))
        topR = sorted(indexed_vehicles, key=lambda x: x[1], reverse=True)[:self.R]
        self.vehicles = {index: value for index, value in topR}

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

        Usage:
            The action format is as follows:    
                (<type of action>, <vehicle>, <request>, <action time>)

        Returns:
            float: The dropoff time of the action.

        """
        return action[3]
    
    def is_dropoff(self, action):
        """Returns True if an action is a dropoff action, False otherwise.

        Args:
            action (tuple): A tuple representing an action.
        
        Usage:
            The action format is as follows:    
                (<type of action>, <vehicle>, <request>, <action time>)

        Returns:
            bool: True if the action is a dropoff action, False otherwise.

        """
        return action[0] == 'Dropoff'
        
    def get_request_time(self, action):
        """Returns the request time of an action.

        Args:
            action (tuple): A tuple representing an action.
        
        Usage:
            The action format is as follows:    
                (<type of action>, <vehicle>, <request>, <action time>)

        Returns:
            float: The request time of the action.

        """
        return self.requests[action[2]][0]
    
    def get_trip_time(self, action):
        """Returns the trip time of an action.

        Args:
            action (tuple): A tuple representing an action.
        
        Usage:
            The action format is as follows:    
                (<type of action>, <vehicle>, <request>, <action time>)

        Returns:
            float: The trip time of the action.

        """
        request = self.requests[action[2]]
        return self.matrix[request[1]][request[2]]
    
    def cost(self, sol):
        """Computes the cost of solution sol.

        Args:
            sol (list): A list of actions representing a solution.

        Usage:
            An action format is as follows:
                (<type of action>, <vehicle>, <request>, <action time>)

        Returns:
            float: The cost of the solution.

        """
        cost = 0
        for action in sol:
            if self.is_dropoff(action):
                cost += self.get_action_time(action) - self.get_request_time(action) - self.get_trip_time(action)
        
        return cost
    
    def result(self, state, action):
        """
        Returns the state that results from executing action in state.
        The action must be one of self.actions(state).
        
        Args:
            state (tuple): A tuple representing a state.
            action (tuple): A tuple representing an action.
            
        Usage:
            The action format is as follows:
                (<type of action>, <vehicle>, <request>, <action time>)
            The state is a tuple where each position is formated as follows:
                (<request state>) if request have not been picked up
                (<request state>, <time>, <vehicle>) any other case
                
        Returns:
            tuple: The resulting state.
            
        """
        car = action[1]
        request = action[2]
        time = action[3]
        requests = State(state[:request] + (((1, time, car,),) if state[request][0] == 0 else ((2, time, car,),)) + state[request + 1:])
        return requests
    
    def actions(self, state):
        """
        Returns the actions that can be executed in the given state.
        
        Args:
            state (tuple): A tuple representing a state.
            
        Usage:
            The state is a tuple where each position is formated as follows:
                (<request state>) if request have not been picked up
                (<request state>, <time>, <vehicle>) any other case
            The return list positions are actions that are formated as follows:
                (<type of action>, <vehicle>, <request>, <action time>)
        
        Returns:
            list: A list of actions that can be executed in the given state.
        """
        # Actions are formated as follows:
        # (<type of action>, <vehicle>, <request>, <action time>)
        actions = []
        # Picks is a list of requests that have not been picked up
        picks = []
        # Drops is a list of tuples (request, vehicle) where request has been picked up and not dropped off
        drops = []
        # Pos is a list of positions of the vehicles
        pos = [0] * self.V
        # Time is a list of times of the vehicles
        time = [0] * self.V
        # Occupation is a list of the occupation of the vehicles
        occupation = [0] * self.V

        for i, r in enumerate(state):
            s = r[0]
            if s == 0:
                # Add request to picks list
                picks.append(i)
            elif s == 1:
                # Add request to drops list
                t = r[1]
                car = r[2]
                drops.append((i, car))
                # Update occupation
                occupation[car] += self.requests[i][3]
                # Update time and position (the vehicle position and time are conditioned by last action executed)
                if t > time[car]:
                    time[car] = t
                    pos[car] = self.requests[i][1]
            else:
                t = r[1]
                car = r[2]
                # Update time and position (the vehicle position and time are conditioned by last action executed)
                if t > time[car]:
                    time[car] = t
                    pos[car] = self.requests[i][2]

        for i in picks:
            # Add pickup action for each vehicle that can pick up the request
            for v in self.vehicles.keys():
                # Vehicle must have enough capacity to pick up the request
                if occupation[v] + self.requests[i][3] <= self.vehicles[v]:
                    t = time[v] + self.matrix[pos[v]][self.requests[i][1]]
                    # The pickup time must be equal or greater than the request time
                    actions.append(('Pickup', v, i, t if t >= self.requests[i][0] else self.requests[i][0]))

        # Add dropoff action to the actions list for each request that has been picked up and not dropped off
        for i, v in drops:
            t = time[v] + self.matrix[pos[v]][self.requests[i][2]]
            actions.append(('Dropoff', v, i, t))
        
        return actions
    
    def goal_test(self, state):
        """
        Returns True if state is a goal state, False otherwise.

        Args:
            state (tuple): A tuple representing a state.

        Usage:
            The state is a tuple where each position is formated as follows:
                (<request state>) if request have not been picked up
                (<request state>, <time>, <vehicle>) any other case
            A goal state is a state where all requests are on state 2.

        Returns:
            bool: True if state is a goal state, False otherwise.
        """
        return all([r[0] == 2 for r in state])

    def path_cost(self, c, state1, action, state2):
        """
        Returns the cost of a solution path that arrives at state2 from state1 via action,
        assuming cost c to get up to state1.
        The action must be one of self.actions(state1).
        
        Args:
            c (float): The cost to get up to state1.
            state1 (tuple): A tuple representing a state.
            action (tuple): A tuple representing an action.
            state2 (tuple): A tuple representing a state.

        Usage:
            The action format is as follows:
                (<type of action>, <vehicle>, <request>, <action time>)
            The state is a tuple where each position is formated as follows:
                (<request state>) if request have not been picked up
                (<request state>, <time>, <vehicle>) any other case

        Returns:
            float: The cost of a solution path that arrives at state2 from state1 via action,
            assuming cost c to get up to state1.
        """
        request = action[2]
        if not self.is_dropoff(action):
            return c + self.get_action_time(action) - self.get_request_time(action)
        else:
            return c + self.get_action_time(action) - state1[request][1] - self.get_trip_time(action)

    def get_best_cost(self, state, car, requests, position, time):
        best_cost = float('inf')
        for perm in itertools.permutations(requests):
            copy = state
            p_copy = position
            t_copy = time
            c_copy = 0
            for i in perm:
                t_copy += self.matrix[p_copy][self.requests[i][2]]
                p_copy = self.requests[i][2]
                action = ('Dropoff', car, i, t_copy)
                aux = self.result(copy, action)
                c_copy = self.path_cost(c_copy, copy, action, aux)
                copy = aux
            if c_copy < best_cost:
                best_cost = c_copy
        return best_cost
    
    def get_ocupation(self, requests):
        return sum([self.requests[i][3] for i in requests])
    
    def get_fastest_dropoff(self, cars, capacity, destination):
        best_time = float('inf')
        for c in cars.keys():
            car = cars[c]
            if self.vehicles[c] >= capacity:
                for perm in itertools.permutations(car[2]):
                    time = car[0]
                    position = car[1]
                    for r in perm:
                        if self.vehicles[c] - self.get_ocupation(perm) >= capacity:
                            break
                        time += self.matrix[position][self.requests[r][2]]
                        position = self.requests[r][2]
                    time += self.matrix[position][destination]
                    if time < best_time:
                        best_time = time
        return best_time
    
    def h(self, node):
        cost_left = 0
        state = node.state
        cars = {}
        for i, r in enumerate(state):
            if r[0] == 1:
                car = r[2]
                if car not in cars:
                    cars[car] = (r[1], self.requests[i][1], (i,))
                elif r[1] > cars[r[2]][0]:
                    cars[car] = (r[1], self.requests[i][1], cars[car][2] + (i,))
                else:
                    cars[car] = (cars[car][0], cars[car][1], cars[car][2] + (i,))
            elif r[0] == 2:
                car = r[2]
                if car not in cars:
                    cars[car] = (r[1], self.requests[i][2], ())
                elif r[1] > cars[r[2]][0]:
                    cars[car] = (r[1], self.requests[i][2], cars[car][2])
        
        for i in cars.keys():
            car = cars[i]
            if car[2]:
                cost_left += self.get_best_cost(state, i, car[2], car[1], car[0])
        
        for i, r in enumerate(state):
            if r[0] == 0:
                if any([c not in cars and self.vehicles[c] >= self.requests[i][3] for c in self.vehicles.keys()]):
                    best_arr_time = self.matrix[0][self.requests[i][1]]
                else:
                    best_arr_time = float('inf')
                for c in cars.keys():
                    car = cars[c]
                    if self.get_ocupation(car[2]) + self.requests[i][3] <= self.vehicles[c]:
                        arr_time = car[0] + self.matrix[car[1]][self.requests[i][1]]
                        if arr_time < best_arr_time:
                            best_arr_time = arr_time
                if best_arr_time == float('inf'):
                    best_arr_time = self.get_fastest_dropoff(cars, self.requests[i][3], self.requests[i][1])
                cost_left += max(best_arr_time - self.requests[i][0], 0)

        return cost_left

    def solve(self):
        """
        Returns a solution to the problem.
        Uses the uniform_cost_search method from the search module.
        
        Returns:
            list: A list of actions representing a solution to the problem.
        """
        return search.astar_search(self, h=self.h, display=True).solution()
