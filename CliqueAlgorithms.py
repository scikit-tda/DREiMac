import numpy as np
import scipy
from scipy import sparse

def get3CliquesBrute(Edges):
    """
    Brute force function to check for all 3 cliques by checking
    mutual neighbors between 3 vertices
    """
    [I, J, V] = [[], [], []]
    NVertices = len(Edges)
    MaxNum = int(NVertices*(NVertices-1)*(NVertices-2)/6)
    J = np.zeros((MaxNum, 3))
    [i, j, k] = [0, 0, 0]
    edgeNum = 0
    for i in range(NVertices):
        for j in Edges[i]:
            if j < i:
                continue
            for k in Edges[j]:
                if k < j or k < i:
                    continue
                if k in Edges[i]:
                    [a, b, c] = sorted([i, j, k])
                    J[edgeNum, :] = [Edges[a][b], Edges[a][c], Edges[b][c]]
                    edgeNum += 1
    J = J[0:edgeNum, :]
    V = np.zeros(J.shape)    
    V[:, 0] = 1
    V[:, 1] = -1
    V[:, 2] = 1
    I = np.zeros(J.shape)
    for k in range(3):
        I[:, k] = np.arange(I.shape[0])
    return (I, J, V)

