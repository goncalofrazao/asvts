import sys
import random

args = sys.argv
# args[0] = map_generator.py
# args[1] = output file name
# args[2] = number of points
# args[3] = max number of requests
# args[4] = max number of vehicles

file = open(args[1], "w")

P = int(args[2])

R = random.randint(1, int(args[3]) + 1)

V = random.randint(1, int(args[4]) + 1)

file.write("P " + str(P) + "\n")

for i in range(P - 1, 0, -1):
    for j in range(i):
        file.write(" " + str(random.randint(1, 100)))
    file.write("\n")

file.write("R " + str(R) + "\n")

for i in range(R):
    file.write(str(random.randint(1, 100)) + " " + str(random.randint(0, P - 1)) + " " + str(random.randint(0, P - 1)) + " " + str(random.randint(0, V)) + "\n")

file.write("V " + str(V) + "\n")

for i in range(V):
    file.write(str(random.randint(1, 10)) + "\n")
    
file.close()