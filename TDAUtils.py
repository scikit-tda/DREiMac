import numpy as np

def add_cocycles(c1, c2, p = 2):
    S = {}
    c = np.concatenate((c1, c2), 0)
    for k in range(c.shape[0]):
        [i, j, v] = c[k, :]
        i, j = max(i, j), min(i, j)
        if not (i, j) in S:
            S[(i, j)] = v
        else:
            S[(i, j)] += v
    cret = np.zeros((len(S), 3))
    cret[:, 0:2] = np.array([s for s in S])
    cret[:, 2] = np.array([np.mod(S[s], p) for s in S])
    cret = np.array(cret[cret[:, -1] > 0, :], dtype = np.int64)
    return cret