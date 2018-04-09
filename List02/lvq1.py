import math



def calculateDistance(instance, prototypes):
    distance = []
    for i,item in enumerate(prototypes):
        distance.append([i,euclidean_distance(instance,item)])
    return distance.sort()

def euclidean_distance(instance, x):
    dist = 0.0
    for i in range(len(x)-1):
        dist += (instance[i] - x[i])**2
    return math.sqrt(dist)


     