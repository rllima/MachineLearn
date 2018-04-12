
import distance as dt


def lvq1(train, lrate, prototypes, epochs):
    print("LVQ1")
    for i in range(epochs):
        rate = lrate * (1.0 - (i/float(epochs)))
        totalError = 0.0
        for instance in train:
            prototype = dt.calculateDistance(instance, prototypes)[0][0]
            for j in range(len(instance)-1):
                error = instance[i] - prototype[i]
                totalError += error**2
                if(prototype[-1] == instance[-1]):
                    prototype[j] += rate * error
                else:
                    prototype[j] -= rate * error

        print('>epoch=%d, lrate=%.3f, error=%.3f' % (i, rate, totalError))
    return prototypes;

def window(prot1, prot2, instance, w):
    di = dt.euclidean_distance(instance, prot1)
    dj = dt.euclidean_distance(instance, prot2)
    if di !=0 and dj != 0:
        mini = min(di/dj,dj/di)
    else:
        mini = 0
    s = ((1-w)/(1+w))

    return (mini>s) 
    
    
def lvq21(train,lrate,prototypes,epochs):
    print("LVQ2.1")
    prots = lvq1(train,lrate,prototypes, epochs)
    for i in range(epochs):
        rate = lrate * (1.0 - (i/float(epochs)))
        totalError = 0.0
        for item in train:
            prots = dt.calculateDistance(item,prototypes)
            mi = prots[0][0]
            mj = prots[1][0]

            if window(mi,mj,item,0.3):
                if mi[-1] != mj[-1]:
                    if mi[-1] == item[-1]:
                        for attr in range(len(mi)-1):
                            error = item[attr] - mi[attr]
                            totalError += error**2
                            mi[attr] += rate * error
                            mj[attr] -= rate * error
                    elif mj[-1] == item[-1]:
                        for attr in range(len(mi)-1):
                            error = item[attr] - mi[attr]
                            totalError += error**2
                            mj[attr] += rate * error
                            mi[attr] -= rate * error
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (i, rate, totalError))
    return prototypes                   

                        
                


        
        


    
    
     