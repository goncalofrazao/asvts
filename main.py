from fleetproblem import FleetProblem

def main():
    fp = FleetProblem(open('input.txt'))
    
    print(fp.travel_time_matrix)
    print(fp.requests)
    print(fp.vehicles)


if __name__ == '__main__':
    main()

