from fleetproblem import FleetProblem

def read_solution(fh):
    solution = []
    lines = fh.readlines()
    for line in lines:
        solution.append((line.split(' ')[0], int(line.split(' ')[1]), int(line.split(' ')[2]), float(line.split(' ')[3])))
    
    return solution

def main():
    fp = FleetProblem(open('input.txt'))
    
    # print(fp.travel_time_matrix)
    # print(fp.requests)
    # print(fp.vehicles)

    # for line in fp.travel_time_matrix:
    #     print(line)

    sol = read_solution(open('output.txt'))

    print(fp.cost(sol))

    


if __name__ == '__main__':
    main()

