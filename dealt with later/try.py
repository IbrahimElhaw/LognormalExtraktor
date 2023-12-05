import numpy as np


def constantSpeed(path, density, multiplier=1):
    lengths = [0]
    for i in range(1,len(path)):
        lengths.append(lengths[i-1]+pow(np.sum(pow(path[i,0:2]-path[i-1,0:2],2)), 0.5))
    lTotal = lengths[-1]
    n = round(lTotal / density + 0.5)
    n *= multiplier
    r = []
    j = 0
    alpha = 0
    r.append(path[0])
    for i in range(1,int(n+1)):
        while (n * lengths[j+1] < i * lTotal): 
            j += 1
        alpha = (lengths[j+1] - i*lTotal/(float)(n)) / (float)(lengths[j+1] -lengths[j])
        r.append(path[j]*alpha + path[j+1]*(1-alpha))
    return np.array(r) #.astype("float32")


print(constantSpeed([2,3,4,5],0.5))