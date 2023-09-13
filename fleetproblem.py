class FleetProblem:
    def __init__(self, fh=None):
        self.P = 0
        self.R = 0
        self.V = 0
        self.travel_time_matrix = []
        self.requests = []
        self.vehicles = []
        if fh:
            self.load(fh)
    
    def matrix_triangulation(self):
        for i in range(self.P):
            for j in range(i + 1, self.P):
                self.travel_time_matrix[j][i] = self.travel_time_matrix[i][j]

    def load(self, fh):
        self.__init__()
        p = 0
        r = 0
        v = 0
        for line in fh:
            if p > 0:
                self.travel_time_matrix[self.P - p - 1] = [0] * (self.P - p) + [int(x) for x in line.split(' ') if x]
                p -= 1
            elif r > 0:
                self.requests.append([int(x) for x in line.split(' ')])
                r -= 1
            elif v > 0:
                self.vehicles.append(int(line))
                v -= 1
            elif line.startswith('P'):
                self.P = int(line[2:])
                self.travel_time_matrix = [[0] * self.P] * self.P
                p = self.P - 1
            elif line.startswith('R'):
                self.R = int(line[2:])
                r = self.R
            elif line.startswith('V'):
                self.V = int(line[2:])
                v = self.V
        
        self.matrix_triangulation()

    # def cost1(self, sol):
    #     cost = 0
    #     for i in sol:
    #         if i[0] == 'Pickup':
    #             tp = i[3]
    #             t = self.requests[i[2]][0]
    #             cost += tp - t
        
    #     return cost

    def get_dropoff_time(self, action):
        return action[3]
    
    def is_dropoff(self, action):
        return action[0] == 'Dropoff'
        
    def get_request_time(self, action):
        return self.requests[action[2]][0]
    
    def get_trip_time(self, action):
        request = self.requests[action[2]]
        return self.travel_time_matrix[request[1]][request[2]]
    
    def cost(self, sol):
        cost = 0
        for action in sol:
            if self.is_dropoff(action):
                cost += self.get_dropoff_time(action) - self.get_request_time(action) - self.get_trip_time(action)
        
        return cost
    

