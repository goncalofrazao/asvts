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

    def load(self, fh):
        self.__init__()
        lines = fh.readlines()
        p = 0
        r = 0
        v = 0
        for line in lines:
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