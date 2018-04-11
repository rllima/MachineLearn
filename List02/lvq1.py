import math


def calculateDistance(instance, prototypes):
    distance = []
    for item in prototypes:
        distance.append([item,euclidean_distance(instance,item)])
    distance.sort(key=lambda tup: tup[1])
    return distance[0][0] 

def euclidean_distance(instance, x):
    dist = 0.0
    for i in range(len(x)-1):
        dist += (instance[i] - x[i])**2
    return math.sqrt(dist)

def train(train, lrate, prototypes, epochs):
    for i in range(epochs):
        rate = lrate * (1.0 - (i/float(epochs)))
        totalError = 0.0
        for instance in train:
            prototype = calculateDistance(instance, prototypes)
            for j in range(len(instance)-1):
                error = instance[i] - prototype[i]
                totalError += error**2
                if(prototype[-1] == instance[-1]):
                    prototype[j] += rate * error
                else:
                    prototype[j] -= rate * error

        print('>epoch=%d, lrate=%.3f, error=%.3f' % (i, rate, totalError))
    return prototypes;

     