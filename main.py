# from fleetproblem import FleetProblem
import sys
from solution import FleetProblem
def read_solution(fh):
    solution = []
    lines = fh.readlines()
    for line in lines:
        solution.append((line.split(' ')[0], int(line.split(' ')[1]), int(line.split(' ')[2]), float(line.split(' ')[3])))
    
    return solution

def main():
    args = sys.argv
    fp = FleetProblem(open(args[1]))
    
    # print(fp.matrix)
    print(fp.requests)
    print(fp.vehicles)

    for line in fp.matrix:
        print(line)
    print(fp.cost([('Dropoff', 0, 0, 60.0), ('Pickup', 0, 0, 20.0)]))
    # sol = read_solution(open('output.txt'))
    # print(fp.cost(sol))
    # fp.init_solve()
    # fp.iterative_solve()
    # if FleetProblem.best_state:
    #     print(FleetProblem.best_state.actions)
    # print('cost: ', FleetProblem.best_cost)

if __name__ == '__main__':
    main()

