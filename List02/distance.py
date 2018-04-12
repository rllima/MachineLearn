import math
def calculateDistance(instance, prototypes):
    distance = []
    for item in prototypes:
        distance.append([item,euclidean_distance(instance,item)])
    distance.sort(key=lambda tup: tup[1])
    return distance

def euclidean_distance(instance, x):
    dist = 0.0
    for i in range(len(x)-1):
        dist += (instance[i] - x[i])**2
    return math.sqrt(dist)