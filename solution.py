__author__ = "Gonçalo Frazão, Gustavo Vítor, and Tiago Costa"
__credits__ = ["Gonçalo Frazão", "Gustavo Vítor", "Tiago Costa"]
__version__ = "0.3.7"
__maintainer__ = "Gonçalo Frazão"
__email__ = "goncalobfrazao@tecnico.ulisboa.pt"
__maintainer__ = "Gustavo Vítor"
__email__ = "gustavovitor@tecnico.ulisboa.pt"
__maintainer__ = "Tiago Costa"
__email__ = "tiagoncosta@tecnico.ulisboa.pt"
__status__ = "Finished"

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
    initial (State): The initial state of the problem instance (all request are in state 0)
"""

import search
import utils
import itertools

class Heap(utils.PriorityQueue):
    """
    A decent implementation of a priority queue using a heap.
    Mega boosted over powered priority queue as utils.py does
    not have.
    """
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

class State():
    """
    This class represents a state of the problem.
    
    Attributes:
        req_state (tuple): A tuple with the state of each request.
        Each request state is represented as a tuple with the following attributes:
            (<state of request>, <time that led to this state>, <car that led to this state>)
            The state of a request can be:
                0: The request has not been picked up yet.
                1: The request has been picked up but not dropped off yet.
                2: The request has been dropped off.
            attributes may be only (<state of request>,) if the request has not been picked up yet.
        cars (dict): A dictionary with the state of each car.
        Each car state is represented as a tuple with the following attributes:
            (<time of last action done>, <current position>, <current occupation>, <occupation needed so far in the path done>)
        car_set (list): A list with the capacities of the cars that have not been used yet.
        The car set is sorted from the highest to the lowest capacities.
    """
    def __init__(self, req_state: tuple, cars: dict, car_set: list):
        self.req_state = req_state
        self.cars = cars
        self.car_set = car_set
    
    def __lt__(self, other):
        """
        The state with higher priority is the closest to the final state
        """
        return sum(i[0] for i in self.req_state) >= sum(i[0] for i in other.req_state)
    
    def __eq__(self, other):
        """
        Two states are equal if all requests are in the same status and reached that
        status at the same time, and the car set is the same
        """
        return tuple(i[:2] for i in self.req_state) == tuple(i[:2] for i in other.req_state) and self.car_set == other.car_set
    
    def __hash__(self):
        return hash(tuple(i[:2] for i in self.req_state) + tuple(self.car_set))

class FleetProblem(search.Problem):

    def __init__(self):
        self.P = 0
        self.R = 0
        self.V = 0
        self.matrix = []
        self.requests = []
        self.cars = []
        self.car_set = []
        self.initial = State(None, None, None)

    def load(self, fh):
        """Loads a problem instance from a file.

        Args:
            fh (file): A file handle to a problem instance.

        Usage:
            The file format is as follows:
                P <number of points>
                <matrix of distances>
                R <number of requests>
                <requests>
                V <number of vehicles>
                <vehicles>

        Side effects:
            Modifies the self.P, self.R, self.V, self.matrix, self.requests, self.cars and self.initial
            attributes for the problem instance.

        """
        p = 0
        r = 0
        v = 0
        for line in fh:
            if p > 0:
                self.matrix[self.P - p - 1] = [0] * (self.P - p) + [float(x) for x in line.split()]
                p -= 1
            elif r > 0:
                self.requests.append([float(line.split()[0])] + [int(x) for x in line.split()[1:]])
                r -= 1
            elif v > 0:
                self.cars.append(int(line))
                v -= 1
            elif line.startswith('P'):
                self.P = int(line[2:])
                self.matrix = [[0] * self.P] * self.P
                p = self.P - 1
            elif line.startswith('R'):
                self.R = int(line[2:])
                r = self.R
            elif line.startswith('V'):
                self.V = int(line[2:])
                v = self.V
        
        self.matrix_triangulation()
        self.car_set = sorted(self.cars, reverse=True)

        self.initial = State(tuple((0,) for _ in range(self.R)), {}, self.car_set)

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
    
    def result(self, state: State, action):
        """Returns the state that results from executing action in state.

        Args:
            state (State): A state representation.
            action (tuple): A tuple representing an action.

        Usage:
            The action format is as follows:
                (<type of action>, <vehicle>, <request>, <action time>)
            The state is a State instace

        Returns:
            tuple: The state that results from executing action in state.

        """
        # Extract information from action
        car = action[1]
        request = action[2]
        time = action[3]

        # Extract information from state
        req_state = state.req_state
        car_set: list = state.car_set.copy()
        cars = state.cars.copy()

        # Apply the action to the state
        # As the state is represented for a tuple with the state of each request
        # and the car capacities that have not been used yet, if the action exceds
        # the car capacity, it is necessary to change the car of the path
        current_occupation = 0
        old_need = 0
        new_need = 0
        if car in cars:
            _, _, current_occupation, old_need = cars[car]
        new_occuption = current_occupation + (self.requests[request][3] if req_state[request][0] == 0 else  (-self.requests[request][3]))
        if new_occuption > old_need:
            if old_need > 0:
                car_set.append(old_need)
                car_set = sorted(car_set, reverse=True)
            new_need_index = first_higher(car_set, new_occuption)
            new_need = car_set.pop(new_need_index)
        else:
            new_need = old_need
        
        position = self.requests[request][1] if req_state[request][0] == 0 else self.requests[request][2]
        cars[car] = (time, position, new_occuption, new_need)

        req_state = req_state[:request] + (((1, time, car,),) if req_state[request][0] == 0 else ((2, time, car,),)) + req_state[request + 1:]

        # In case of 1 vehicle per request, the best case is the immediate
        # dropoff of the request, so the result returns the state with the
        # dropoff action already done
        req_missing = [i for i in req_state if i[0] == 0]
        cap_missing = sorted([self.requests[i][3] for i, r in enumerate(req_state) if r[0] == 0], reverse=True)
        new_state = State(req_state, cars, car_set)

        if action[0] == 'Pickup' and self.R <= self.V and len(req_missing) <= len(car_set) and all(i <= j for i, j in zip(cap_missing, car_set)):
            return self.result(new_state, ('Dropoff', car, request, time + self.get_trip_time(action)))
        
        return State(req_state, cars, car_set)
    
    def actions(self, state: State):
        """Returns the actions that can be executed in state.

        Args:
            state (State): A representation of a state.

        Usage:
            The state is a State instace
            An action format is as follows:
                (<type of action>, <vehicle>, <request>, <action time>)

        Returns:
            list: A list of actions that can be executed in state.

        """
        actions = []

        for i, r in enumerate(state.req_state):
            s = r[0]
            if s == 0:
                if state.car_set and self.requests[i][3] <= state.car_set[0]:
                    t = self.matrix[0][self.requests[i][1]]
                    actions.append(('Pickup', len(state.cars), i, max(t, self.requests[i][0])))
                for v in state.cars:
                    car = state.cars[v]
                    # car = (time, position, current occupation, occupation needed so far)
                    if car[2] + self.requests[i][3] <= car[3] or (state.car_set and car[2] + self.requests[i][3] <= state.car_set[0]):
                        t = car[0] + self.matrix[car[1]][self.requests[i][1]]
                        actions.append(('Pickup', v, i, max(t, self.requests[i][0])))
            elif s == 1:
                c = r[2]
                car = state.cars[c]
                t = car[0] + self.matrix[car[1]][self.requests[i][2]]
                actions.append(('Dropoff', c, i, t))
        
        return actions
    
    def goal_test(self, state):
        """
        Returns True if state is a goal state, False otherwise.

        Args:
            state (State): A representation of a state.

        Usage:
            The state is a State instace
            A goal state is a state where all requests are on state 2.

        Returns:
            bool: True if state is a goal state, False otherwise.
        """
        return all(r[0] == 2 for r in state.req_state)

    def path_cost(self, c, state1, action, state2):
        """
        Returns the cost of a solution path that arrives at state2 from state1 via action,
        assuming cost c to get up to state1.
        The action must be one of self.actions(state1).
        
        Args:
            c (float): The cost to get up to state1.
            state1 (State): A representation of a state.
            action (tuple): A tuple representing an action.
            state2 (State): A representation of a state.

        Usage:
            The action format is as follows:
                (<type of action>, <vehicle>, <request>, <action time>)
            The state is a State instace

        Returns:
            float: The cost of a solution path that arrives at state2 from state1 via action,
            assuming cost c to get up to state1.
        """
        request = action[2]
        if not self.is_dropoff(action):
            return c + self.get_action_time(action) - self.get_request_time(action)
        else:
            return c + self.get_action_time(action) - state1.req_state[request][1] - self.get_trip_time(action)

    def get_best_cost(self, state, car, requests, position, time):
        """Calculates the best dropoff order for a car.

        Args:
            state (State): A representation of a state.
            car (int): The car index.
            requests (tuple): The requests to dropoff.
            position (int): The current position of the car.
            time (float): The current time of the car.

        Usage:
            The state is a State instace
            The requests is a tuple of request indexs to dropoff
            The position is the current point of the car
            The time is the current time of the car

        Returns:
            float: The cost of the best dropoff order for a car.
        """
        best_cost = float('inf')
        # Simulate all the dropoff orders possible
        for perm in itertools.permutations(requests):
            copy = state
            p_copy = position
            t_copy = time
            c_copy = 0
            # Calculate the final cost of the dropoff order
            # and save the best cost
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
    
    def get_fastest_dropoff(self, cars, requests, capacity, destination):
        """
        Calculates the fastest car to get a minimum capacity and arrive at a destination.
        
        Args:
            cars (dict): The cars in the state.
            requests (dict): The requests pending in each car.
            capacity (int): The minimum capacity needed.
            destination (int): The destination point.

        Usage:
            The cars is a dict of cars in the state
            Each car is a tuple of (time, position, current occupation, occupation needed so far)
            The requests is a dict of request indexs pending in each car
            The capacity is the minimum capacity at the destination
            The destination is the destination point

        Returns:
            float: The time of the arrival of the fastest car at destination.
        """
        best_time = float('inf')
        # Simulate all dropoff orders for each car
        for c in cars:
            car = cars[c]
            max_capacity = car[3]
            if max_capacity >= capacity:
                for perm in itertools.permutations(requests[c]):
                    time = car[0]
                    position = car[1]
                    ocupation = car[2]
                    # Calculate the time to arrive at destination with the dropoff order permutation
                    for r in perm:
                        if max_capacity - ocupation >= capacity:
                            break
                        time += self.matrix[position][self.requests[r][2]]
                        position = self.requests[r][2]
                        ocupation -= self.requests[r][3]
                    time += self.matrix[position][destination]
                    if time < best_time:
                        best_time = time
        return best_time
    
    def h(self, node):
        """
        Returns the heuristic value for a given state.
        The heuristic value is the minimum cost left possible
        to get a solution.
        Calculates each pickup pending minimum delay and the best
        dropoff order for each car pending requests.
        
        Args:
            node (Node): A search node.
            
        Usage:
            The node is a Node instace
            
        Returns:
            float: The heuristic value for a given state.
        """
        cost_left = 0
        state = node.state
        cars = {}
        
        # Calculate pending requests for each car
        # cars = {car_index: (request 1 to dropoff, request 2 to dropoff, ...)}
        for i, r in enumerate(state.req_state):
            if r[0] == 1:
                car = r[2]
                if car not in cars:
                    cars[car] = (i,)
                else:
                    cars[car] = cars[car] + (i,)
        
        # Calculate best order to dropoff requests for each car

        # get_best_cost does all the permutations of the requests
        #  to dropoff and returns the cost of the best permutation
        for i in cars:
            cost_left += self.get_best_cost(state, i, cars[i], state.cars[i][1], state.cars[i][0])

        # Search the first car to arrive at each request and calculate the cost so far
        
        # The cost of each waiting pickup is the time of the request 
        # minus the time of the first car to arrive at the pickup point
        
        # If no request has capacity to pickup, the get_fastest_dropoff 
        # returns the arrival at pickup point of the car that can get 
        # fastest the capacity to pickup
        for i, r in enumerate(state.req_state):
            if r[0] == 0:
                if state.car_set and self.requests[i][3] <= state.car_set[0]:
                    best_arr_time = self.matrix[0][self.requests[i][1]]
                else:
                    best_arr_time = float('inf')
                    for c in state.cars:
                        car = state.cars[c]
                        if car[2] + self.requests[i][3] <= car[3] or (state.car_set and car[2] + self.requests[i][3] <= state.car_set[0]):
                            arr_time = car[0] + self.matrix[car[1]][self.requests[i][1]]
                            if arr_time < best_arr_time:
                                best_arr_time = arr_time
                    if best_arr_time == float('inf'):
                        best_arr_time = self.get_fastest_dropoff(state.cars, cars, self.requests[i][3], self.requests[i][1]) - 1
                cost_left += max(best_arr_time - self.requests[i][0], 0)

        return cost_left

    def solve(self):
        """
        Returns a solution to the problem.
        Uses the astar method from the search module.
        
        Returns:
            list: A list of actions representing a solution to the problem.
        """
        # Get solution
        last_node = search.astar_search(self, h=self.h, display=True)
        solution_before_conversion = last_node.solution()

        # Convert solution cars to the original cars
        conversion = {}
        for i in solution_before_conversion:
            car = i[1]
            if car not in conversion:
                car_capacity = last_node.state.cars[car][3]
                car_index = self.cars.index(car_capacity)
                self.cars[car_index] = -1
                conversion[car] = car_index
        final_solution = []
        for i in solution_before_conversion:
            final_solution.append((i[0], conversion[i[1]], i[2], i[3]))
        
        # If solution only has pickups add dropoff action for each one
        if last_node.action[0] == 'Pickup':
            for i in range(len(final_solution)):
                final_solution.append(('Dropoff', final_solution[i][1], final_solution[i][2], final_solution[i][3] + self.get_trip_time(final_solution[i])))
        return final_solution

def first_higher(list: list, x: int) -> int:
    """
    Returns the index of the lowest element in a list that is higher or equal to x.
    If no element is higher than x returns -1.

    Args:
        list (list): A list of integers descendent sorted.
        x (int): An integer.

    Usage:
        The list is a list of integers descendent sorted.
        The x is an integer

    Returns:
        int: The index of the lowest element in a list that is higher or equal to x.
        If no element is higher than x returns -1.
    """
    for i in range(len(list)):
        if list[i] < x:
            return i - 1
    return -1
