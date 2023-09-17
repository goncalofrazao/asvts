from fleetproblem import FleetProblem
import sys

def read_solution(fh):
    solution = []
    lines = fh.readlines()
    for line in lines:
        solution.append((line.split(' ')[0], int(line.split(' ')[1]), int(line.split(' ')[2]), float(line.split(' ')[3])))
    
    return solution

def main():
    args = sys.argv
    fp = FleetProblem(open(args[1]))
    
    # print(fp.travel_time_matrix)
    # print(fp.requests)
    # print(fp.vehicles)

    # for line in fp.travel_time_matrix:
    #     print(line)

    # sol = read_solution(open('output.txt'))
    # print(fp.cost(sol))
    # fp.init_solve()
    fp.iterative_solve()
    if FleetProblem.best_state:
        print(FleetProblem.best_state.actions)
    print('cost: ', FleetProblem.best_cost)
    print('counter: ', FleetProblem.counter)

if __name__ == '__main__':
    main()

