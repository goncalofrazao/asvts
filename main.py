# from fleetproblem import FleetProblem
from solution import *
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
    
    # print(fp.matrix)
    # print(fp.requests)
    # print(fp.vehicles)
    # for line in fp.matrix:
    #     print(line)
    # print(fp.cost([('Dropoff', 0, 0, 60.0), ('Pickup', 0, 0, 20.0)]))

    # sol = read_solution(open('output.txt'))
    # print(fp.cost(sol))
    # fp.init_solve()
    # fp.iterative_solve()
    # if FleetProblem.best_state:
    #     print(FleetProblem.best_state.actions)
    # print('cost: ', FleetProblem.best_cost)

    # state = State(fp.requests, len(fp.vehicles))
    # state2 = fp.result(state, ('Pickup', 0, 0, 20.0))
    # state3 = fp.result(state2, ('Dropoff', 0, 0, 60.0))
    # print("########State########")
    # print(state.pickups)
    # print(state.dropoffs)
    # print(state.path)
    # for i in state.vehicles:
    #     print(i.ocuppation, i.time, i.position)
    # print("########State2########")
    # print(state2.pickups)
    # print(state2.dropoffs)
    # print(state2.path)
    # for i in state2.vehicles:
    #     print(i.ocuppation, i.time, i.position)
    # print("########State3########")
    # print(state3.pickups)
    # print(state3.dropoffs)
    # print(state3.path)
    # for i in state3.vehicles:
    #     print(i.ocuppation, i.time, i.position)

    state = State(fp.requests, len(fp.vehicles))
    actions = fp.actions(state)
    print(actions)
    while actions:
        state = fp.result(state, actions[0])
        actions = fp.actions(state)
        print(actions)
    print(fp.goal_test(state))


if __name__ == '__main__':
    main()

