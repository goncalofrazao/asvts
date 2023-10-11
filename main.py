from solution import *
import sys

import ast

def read_solution(f):
    # Read the contents of the file
    contents = f.read()

    # Use ast.literal_eval to convert the string to a list
    return ast.literal_eval(contents)


def main():
    args = sys.argv
    if len(args) == 2:
        fp = FleetProblem(open(args[1]))

        sol = fp.solve()
        print(fp.cost(sol))

        # solution = sol.solution()
        # path = sol.path()
        # for i in path:
        #     print(i.state.requests)
        # print()
        # print('Path')
        # for i in path:
        #     print(i.path_cost)
        # print()
        # print('Solution')
        # for i in solution:
        #     print(i)
        # print()
        # print('Cost')
        # print(fp.cost(solution))
        # print()
        # print('Requests')
        # for i in fp.requests:
        #     print(i)
        # print()
        # print('Matrix')
        # for line in fp.matrix:
        #     print(line)

    if len(args) == 3:
        list = read_solution(open(args[1]))
        datfile = args[1].replace('.plan', '.dat')
        fp = FleetProblem(open(datfile))
        print(fp.cost(list))

if __name__ == '__main__':
    main()

