import numpy as np

def gs(v):
    w = v.copy() # copy basis

    # start gram-schmidt process to get orthogonal basis
    for i in range(1,len(w)):
        for j in range(i):
            w[i] = w[i] - np.dot( np.dot( w[i],w[j] ) / np.dot( w[j],w[j] ), w[j] ) # if w[i] is w[2], w[j] is w[1]

    # Convert to unit vectors
    w = [vec/np.linalg.norm(vec) for vec in w]

    return w



# v1, v2, v3, v4 = [1,-1,0], [1,2,1], [0,1,1]
# v = [v1,v2,v3]
print(gs([[-1,1,0,2],
          [1,-1,-1,0],
          [2,0,1,1],
          [1,0,0,1]]))




    

