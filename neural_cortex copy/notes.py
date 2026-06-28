C = A[:, None] - B[None, :]
C[i,j]=A[i]-B[j]


C=A[None:]-B[:,None]
C[i,j]=B[i]-A[j]