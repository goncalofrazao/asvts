# from fleetproblem import FleetProblem
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

        path = fp.solve()
        print(path)
        print(fp.cost(path))

    if len(args) == 3:
        list = read_solution(open(args[1]))
        datfile = args[1].replace('.plan', '.dat')
        fp = FleetProblem(open(datfile))
        print(fp.cost(list))

if __name__ == '__main__':
    main()

