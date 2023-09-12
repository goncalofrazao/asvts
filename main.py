
def main():
    P = 0
    R = 0
    V = 0
    p = 0
    r = 0
    v = 0

    travel_time_matrix = []
    requests = []
    vehicles = []
    with open('input.txt') as fp:
        lines = fp.readlines()
        for line in lines:
            if p > 0:
                travel_time_matrix[P - p - 1] = [0] * (P - p) + [int(x) for x in line.split(' ') if x]
                p -= 1
            elif r > 0:
                requests.append([int(x) for x in line.split(' ')])
                r -= 1
            elif v > 0:
                vehicles.append(int(line))
                v -= 1
            elif line.startswith('P'):
                P = int(line[2:])
                travel_time_matrix = [[0] * P] * P
                p = P - 1
            elif line.startswith('R'):
                R = int(line[2:])
                r = R
            elif line.startswith('V'):
                V = int(line[2:])
                v = V
    
    print(travel_time_matrix)
    print(requests)
    print(vehicles)


                

                



if __name__ == '__main__':
    main()

