import numpy as np

def gs(v):
    w = v.copy() # copy basis

    # start gram-schmidt process to get orthogonal basis
    for i in range(1,len(v)):
        for j in range(i):
            w[i] = w[i] - np.dot( np.dot( v[i],w[j] ) / np.dot( w[j],w[j] ), w[j] )

    # Get orthonormal basis
    w = [vec/np.linalg.norm(vec) for vec in w]

    return w



v1, v2, v3 = [1,-1,0], [1,2,1], [0,1,1]
v = [v1,v2,v3]
print(gs(v))




    

