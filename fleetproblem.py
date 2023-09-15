class Vehicle:
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity
        self.requests = {}
        self.time = 0
        self.position = 0
    
    def passengers(self):
        passengers = 0
        for request in self.requests.values():
            passengers += request[3]
        return passengers
    
    def calculate_pickup_time(self, request, matrix):
        if self.time + matrix[self.position][request[1]] >= request[0]:
            return self.time + matrix[self.position][request[1]]
        else:
            return request[0]

    def calculate_dropoff_time(self, request, matrix):
        return self.time + matrix[self.position][request[2]]

    def dropoffs(self, matrix):
        dropoffs = []
        for key, request in self.requests.items():
            dropoffs.append(('Dropoff', self.id, key, self.calculate_dropoff_time(request, matrix)))
        return dropoffs
    
    def drop(self, action):
        request = self.requests[action[2]]
        self.position = request[2]
        self.time = action[3]
        self.requests.pop(action[2])

    def pick(self, action, request):
        self.position = request[1]
        self.time = action[3]
        self.requests[action[2]] = request

    def copy(self):
        new_vehicle = Vehicle(self.id, self.capacity)
        new_vehicle.requests = self.requests.copy()
        new_vehicle.time = self.time
        new_vehicle.position = self.position
        return new_vehicle

class State:
    def __init__(self, requests, vehicles, matrix):
        self.requests = {i: request for i, request in enumerate(requests)}
        self.vehicles = [Vehicle(i, capacity) for i,capacity in enumerate(vehicles)]
        self.actions = []
        self.matrix = matrix

    def get_possible_actions(self):
        actions = []
        for i, vehicle in enumerate(self.vehicles):
            actions += vehicle.dropoffs(self.matrix)
            for key, request in self.requests.items():
                if vehicle.passengers() + request[3] <= vehicle.capacity:
                    actions.append(('Pickup', i, key, vehicle.calculate_pickup_time(request, self.matrix)))
        return actions
    
    def update(self, action):
        vehicle = self.vehicles[action[1]]
        if action[0] == 'Dropoff':
            vehicle.drop(action)
        elif action[0] == 'Pickup':
            vehicle.pick(action, self.requests[action[2]])
            self.requests.pop(action[2])

        self.actions.append(action)
    
    def copy(self):
        new_state = State([], [], [])
        new_state.requests = self.requests.copy()
        new_state.vehicles = [vehicle.copy() for vehicle in self.vehicles]
        new_state.actions = self.actions.copy()
        new_state.matrix = self.matrix.copy()
        return new_state

class FleetProblem:
    best_cost = -1
    best_state = None
    counter = 0

    def __init__(self, fh=None):
        self.P = 0
        self.R = 0
        self.V = 0
        self.matrix = []
        self.requests = []
        self.vehicles = []
        if fh:
            self.load(fh)
    
    def init_solve(self):
        state = State(self.requests, self.vehicles, self.matrix)
        self.solve(state)
    
    def heuristic(self, action):
        return action[3]

    def solve(self, state):
        this_iteration_cost = self.cost(state.actions)
        if FleetProblem.best_cost != -1 and this_iteration_cost > FleetProblem.best_cost:
            return
        
        actions = state.get_possible_actions()
        if actions:
            # actions = sorted(actions, key=lambda action: self.heuristic(action))
            # print(actions)
            # print()
            for action in actions:
                # self.solve(state.copy().update(action))
                new_state = state.copy()
                new_state.update(action)
                self.solve(new_state)
        elif this_iteration_cost <= FleetProblem.best_cost or FleetProblem.best_cost == -1:
            # print(state.actions, this_iteration_cost)
            # print('\n')
            FleetProblem.counter += 1
            FleetProblem.best_cost = this_iteration_cost
            FleetProblem.best_state = state

    def matrix_triangulation(self):
        for i in range(self.P):
            for j in range(i + 1, self.P):
                self.matrix[j][i] = self.matrix[i][j]

    def load(self, fh):
        self.__init__()
        p = 0
        r = 0
        v = 0
        for line in fh:
            if p > 0:
                self.matrix[self.P - p - 1] = [0] * (self.P - p) + [int(x) for x in line.split(' ') if x]
                p -= 1
            elif r > 0:
                self.requests.append([int(x) for x in line.split(' ')])
                r -= 1
            elif v > 0:
                self.vehicles.append(int(line))
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

    def get_dropoff_time(self, action):
        return action[3]
    
    def is_dropoff(self, action):
        return action[0] == 'Dropoff'
        
    def get_request_time(self, action):
        return self.requests[action[2]][0]
    
    def get_trip_time(self, action):
        request = self.requests[action[2]]
        return self.matrix[request[1]][request[2]]
    
    def cost(self, sol):
        cost = 0
        for action in sol:
            if self.is_dropoff(action):
                cost += self.get_dropoff_time(action) - self.get_request_time(action) - self.get_trip_time(action)
        
        return cost
    
