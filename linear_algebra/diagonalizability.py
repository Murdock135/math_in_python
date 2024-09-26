# this program checks the diagonalizability of any matrix

import numpy as np

A = np.array([[2,3],
             [1,4]])

print(A.shape)

w,v = np.linalg.eig(A)


print("eigenvalues: ",w)
print("eigenvectors: ",v)

X, X_inv = v, np.linalg.inv(v)

D = np.round(X_inv@A@X,2)

print("D = ",D)

B = np.round(X_inv@np.sqrt(D)@X,2)

print("B = ",B)

print("B*B = ", B@B)
