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
        fp = FleetProblem()
        fp.load(open(args[1]))

        sol = fp.solve()
        print(fp.cost(sol))

    if len(args) == 3:
        list = read_solution(open(args[1]))
        datfile = args[1].replace('.plan', '.dat')
        fp = FleetProblem(open(datfile))
        print(fp.cost(list))

if __name__ == '__main__':
    main()

