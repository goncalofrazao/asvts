import search

class FleetProblem(search.Problem):

    def __init__(self, fh=None):
        self.P = 0
        self.R = 0
        self.V = 0
        self.matrix = []
        self.requests = []
        self.vehicles = []
        if fh:
            self.load(fh)

    def load(self, fh):
        self.__init__()
        p = 0
        r = 0
        v = 0
        for line in fh:
            if p > 0:
                self.matrix[self.P - p - 1] = [0] * (self.P - p) + [float(x) for x in line.split(' ') if x]
                p -= 1
            elif r > 0:
                self.requests.append([float(line.split(' ')[0])] + [int(x) for x in line.split(' ')[1:]])
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
    
    def matrix_triangulation(self):
        for i in range(self.P):
            for j in range(i + 1, self.P):
                self.matrix[j][i] = self.matrix[i][j]
    
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