import numpy as np

A = np.array([[0, 1],
    [2, 3],
    [4, 5]])



cols = len(A[0])
for col in range(cols):
    print("col:",col)
    for current_row in range(len(A)):
        for other_row in range(len(A)):
            if other_row != current_row and A[current_row,col] != 0:
                print("A before row op: ",A)
                alpha = A[other_row,col]/A[current_row,col]

                A[other_row,col] = A[other_row,col] - A[current_row,col]*alpha
                print("A after row op: ",A)

print("Final A: ",A)



                

