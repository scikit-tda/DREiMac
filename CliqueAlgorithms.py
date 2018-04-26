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

def get3CliquesNetworkx(Edges):
    import networkx as nx
    import itertools
    G = nx.Graph()
    edgeslist = []
    for i, edgeDict in enumerate(Edges):
        for j in edgeDict:
            if i < j:
                edgeslist.append((i, j))
    G.add_edges_from(edgeslist)
    cliques = list(nx.find_cliques(G))
    #print(cliques)
    cliques3 = []
    for c in cliques:
        c = sorted(c)
        cliques3.append(c)
        for c3 in itertools.combinations(c, 3):
            cliques3.append(c3)
    cliques3 = np.array(cliques3)
    print(cliques3.size)



def testCliqueTimes():
    """
    Compare brute force to networkx Bron Kerbosch
    """
    import time
    np.random.seed(1)
    N = 1000
    D = np.random.rand(N, N)
    D = 0.5*(D + D.T)
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    D[I == J] = 100
    I = I[D < 0.3]
    J = J[D < 0.3]
    #Make a list of edges for fast lookup
    Edges = []
    for i in range(N):
        Edges.append({})
    for i, (a, b) in enumerate(zip(I, J)):
        Edges[a][b] = i
        Edges[b][a] = i
    
    """
    tic = time.time()
    (I, J, V) = get3CliquesBrute(Edges)
    print("Elapsed time brute force %i Edges: %g"%(I.size, time.time()-tic))
    """

    tic = time.time()
    get3CliquesNetworkx(Edges)
    print("Elapsed time networkx force %i Edges: %g"%(I.size, time.time()-tic))


if __name__ == "__main__":
    testCliqueTimes()
