
import distance as dt


def lvq1(train, lrate, prototypes, epochs):
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
                            errorMi = item[attr] - mi[attr]
                            errorMj = item[attr] - mj[attr]
                            totalError += errorMi**2
                            mi[attr] += rate * errorMi
                            mj[attr] -= rate * errorMj
                    elif mj[-1] == item[-1]:
                        for attr in range(len(mi)-1):
                            errorMi = item[attr] - mi[attr]
                            errorMj = item[attr] - mj[attr]
                            totalError += errorMj**2
                            mj[attr] += rate * errorMj
                            mi[attr] -= rate * errorMi
    return prototypes

def lvq3(train,lrate,prototypes,epochs):
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
                            errorMi = item[attr] - mi[attr]
                            errorMj = item[attr] - mj[attr]
                            totalError += errorMi**2
                            mi[attr] += rate * errorMi
                            mj[attr] -= rate * errorMj
                    elif mj[-1] == item[-1]:
                        for attr in range(len(mi)-1):
                            errorMi = item[attr] - mi[attr]
                            errorMj = item[attr] - mj[attr]
                            totalError += errorMj**2
                            mj[attr] += rate * errorMj
                            mi[attr] -= rate * errorMi
                elif mi[-1] == mj[-1]:
                    for attr in range(len(mi)-1):
                            errorMi = item[attr] - mi[attr]
                            errorMj = item[attr] - mj[attr]
                            totalError += (errorMj+errorMi)**2
                            mi[attr] += (0.3*rate) * errorMi
                            mj[attr] -= (0.3*rate) * errorMj
    return prototypes                                      

                        
                


        
        


    
    
     