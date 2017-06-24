"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: To wrap around Ripser to compute persistence diagrams
"""
import subprocess
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from Hodge import *

def plotDGM(dgm, color = 'b', sz = 20, label = 'dgm', axcolor = np.array([0.0, 0.0, 0.0]), marker = None):
    if dgm.size == 0:
        return
    # Create Lists
    # set axis values
    axMin = np.min(dgm)
    axMax = np.max(dgm)
    axRange = axMax-axMin
    a = max(axMin - axRange/5, 0)
    b = axMax+axRange/5
    # plot line
    plt.plot([a, b], [a, b], c = axcolor, label = 'none')
    plt.hold(True)
    # plot points
    if marker:
        H = plt.scatter(dgm[:, 0], dgm[:, 1], sz, color, marker, label=label, edgecolor = 'none')
    else:
        H = plt.scatter(dgm[:, 0], dgm[:, 1], sz, color, label=label, edgecolor = 'none')
    # add labels
    plt.xlabel('Time of Birth')
    plt.ylabel('Time of Death')
    return H

def plotDGMAx(ax, dgm, color = 'b', sz = 20, label = 'dgm'):
    if dgm.size == 0:
        return
    axMin = np.min(dgm)
    axMax = np.max(dgm)
    axRange = axMax-axMin;
    ax.scatter(dgm[:, 0], dgm[:, 1], sz, color,label=label)
    ax.hold(True)
    ax.plot([axMin-axRange/5,axMax+axRange/5], [axMin-axRange/5, axMax+axRange/5],'k');
    ax.set_xlabel('Time of Birth')
    ax.set_ylabel('Time of Death')

def plot2DGMs(P1, P2, l1 = 'Diagram 1', l2 = 'Diagram 2'):
    plotDGM(P1, 'r', 10, label = l1)
    plt.hold(True)
    plt.plot(P2[:, 0], P2[:, 1], 'bx', label = l2)
    plt.legend()
    plt.xlabel("Birth Time")
    plt.ylabel("Death Time")

def savePD(filename, I):
    if os.path.exists(filename):
        os.remove(filename)
    fout = open(filename, "w")
    for i in range(I.shape[0]):
        fout.write("%g %g"%(I[i, 0], I[i, 1]))
        if i < I.shape[0]-1:
            fout.write("\n")
    fout.close()

#Wrap around Dionysus's bottleneck distance after taking the log
def getInterleavingDist(PD1, PD2):
    savePD("PD1.txt", np.log(PD1))
    savePD("PD2.txt", np.log(PD2))
    proc = subprocess.Popen(["./bottleneck", "PD1.txt", "PD2.txt"], stdout=subprocess.PIPE)
    lnd = float(proc.stdout.readline())
    return np.exp(lnd) - 1.0 #Interleaving dist is 1 + eps

def getBottleneckDist(PD1, PD2):
    savePD("PD1.txt", PD1)
    savePD("PD2.txt", PD2)
    proc = subprocess.Popen(["./bottleneck", "PD1.txt", "PD2.txt"], stdout=subprocess.PIPE)
    return float(proc.stdout.readline())

def doRipsFiltrationDM(D, maxHomDim, thresh = -1, coeff = 2):
    N = D.shape[0]
    #Step 1: Extract and output lower triangular distance matrix
    fout = open("DLower.txt", "w")
    for i in range(1, N):
        for j in range(0, i):
            fout.write("%g "%D[i, j])
    fout.close()

    #Step 2: Call ripser
    callThresh = 2*np.max(D)
    if thresh > 0:
        callThresh = thresh
    if coeff > 2:
        proc = subprocess.Popen(["ripser/ripser-coeff", "--dim", "%i"%maxHomDim, "--threshold", "%g"%callThresh, "--modulus", "%i"%coeff, "DLower.txt"], stdout=subprocess.PIPE)
    else:
        proc = subprocess.Popen(["ripser/ripser", "--dim", "%i"%maxHomDim, "--threshold", "%g"%callThresh, "DLower.txt"], stdout=subprocess.PIPE)
    #stdout = proc.communicate()[0]
    PDs = []
    while True:
        output=proc.stdout.readline()
        if (output == b'' or output == '') and proc.poll() is not None:
            break
        if output:
            s = output.strip()
            if output[0:4] == b"dist":
                continue
            elif output[0:4] == b"valu":
                continue
            elif output[0:4] == b"pers":
                if len(PDs) > 0:
                    PDs[-1] = np.array(PDs[-1])
                PDs.append([])
            else:
                s = s.replace("[", "")
                s = s.replace("]", "")
                s = s.replace("(", "")
                s = s.replace(")", "")
                s = s.replace(" ", "")
                fields = s.split(",")
                b = float(fields[0])
                d = -1
                if len(fields[1]) > 0:
                    d = float(fields[1])
                PDs[-1].append([b, d])
        rc = proc.poll()
    PDs[-1] = np.array(PDs[-1])
    return PDs

def parseCocycle(s):
    s2 = "" + s
    for c in ["]", "[", ",", "{", "}"]:
        s2 = s2.replace(c, "")
    s2 = s2.replace(":", " ")
    cocycle = [int(c) for c in s2.split()]
    cocycle = np.array(cocycle)
    cocycle = np.reshape(cocycle, [len(cocycle)/3, 3])
    return cocycle


def doRipsFiltrationDMCocycles(D, maxHomDim, thresh = -1, coeff = 2):
    import json
    N = D.shape[0]
    #Step 1: Extract and output lower triangular distance matrix
    fout = open("D.txt", "w")
    for i in range(0, N):
        for j in range(0, N):
            fout.write("%g "%D[i, j])
        if i < N-1:
            fout.write("\n")
    fout.close()

    #Step 2: Call ripser
    callThresh = 2*np.max(D)
    if thresh > 0:
        callThresh = thresh
    proc = subprocess.Popen(["ripser/ripser-representatives", "--format", "distance", "--dim", "%i"%maxHomDim, "--threshold", "%g"%callThresh, "--modulus", "%i"%coeff, "D.txt"], stdout=subprocess.PIPE)
    #stdout = proc.communicate()[0]
    PDs = []
    AllCocycles = []
    while True:
        output=proc.stdout.readline()
        if (output == b'' or output == '') and proc.poll() is not None:
            break
        if output:
            s = output.strip()
            if output[0:4] == b"dist":
                continue
            elif output[0:4] == b"valu":
                continue
            elif output[0:4] == b"pers":
                if len(PDs) > 0:
                    PDs[-1] = np.array(PDs[-1])
                PDs.append([])
                AllCocycles.append([])
            else:
                s = s.split(": ")
                if len(s) > 1:
                    [s, s1] = s
                    c = parseCocycle(s1)
                    AllCocycles[-1].append(c)
                else:
                    s = s[0]
                s = s.replace("[", "")
                s = s.replace("]", "")
                s = s.replace("(", "")
                s = s.replace(")", "")
                s = s.replace(" ", "")
                fields = s.split(",")
                b = float(fields[0])
                d = -1
                if len(fields[1]) > 0:
                    d = float(fields[1])
                PDs[-1].append([b, d])
        rc = proc.poll()
    PDs[-1] = np.array(PDs[-1])
    return (PDs, AllCocycles)


def getSSM(X):
    XSqr = np.sum(X**2, 1)
    D = XSqr[:, None] + XSqr[None, :] - 2*X.dot(X.T)
    D[D < 0] = 0 #Numerical precision
    D = np.sqrt(D)
    return D

#Wrapper around Uli Bauer's ripser code
def doRipsFiltration(X, maxHomDim, thresh = -1, coeff = 2, cocycles = False):
    #Compute all pairwise distances assuming Euclidean
    D = getSSM(X)
    if cocycles:
        return doRipsFiltrationDMCocycles(D, maxHomDim, thresh, coeff)
    else:
        return doRipsFiltrationDM(D, maxHomDim, thresh, coeff)

def doRipsFiltrationDionysus(X, coeff = 2, thresh = -1):
    #rips-pairwise-cohomology points.txt -m 1 -b points.bdry -c points -v points.vrt -d points.dgm
    fout = open("points.txt", "w")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            fout.write("%g"%X[i, j])
            if j < X.shape[1]-1:
                fout.write(" ")
        if i < X.shape[0]-1:
            fout.write("\n")
    fout.close()

    D = getSSM(X)
    callThresh = 2*np.max(D)
    if thresh > 0:
        callThresh = thresh

    subprocess.call(["./rips-pairwise-cohomology", "points.txt", "-p", "%i"%coeff, "-m", "%g"%callThresh, "-b", "points.bdry", "-c", "points", "-v points.vrt", "-d", "points.dgm"])

def plotTriangles(X, A, B, C):
    plt.hold(True)
    ax = plt.gca()
    for i in range(len(A)):
        poly = [X[A[i], :], X[B[i], :], X[C[i], :]]
        ax.add_patch(Polygon(np.array(poly), linestyle='solid', color='#00FF00', alpha=0.05))

def drawLineColored(X, C):
    plt.hold(True)
    for i in range(X.shape[0]-1):
        plt.plot(X[i:i+2, 0], X[i:i+2, 1], c=C[i, :], lineWidth = 3)

def plotCocycle(X, cocycle, thresh, drawTriangles = False):
    XSqr = np.sum(X**2, 1)
    D = XSqr[:, None] + XSqr[None, :] - 2*X.dot(X.T)
    D[D < 0] = 0 #Numerical precision
    D = np.sqrt(D)

    plt.hold(True)
    ax = plt.gca()
    #Plot all edges under the threshold
    N = X.shape[0]
    t = np.linspace(0, 1, 10)
    c = plt.get_cmap('Greys')
    C = c(np.array(np.round(np.linspace(0, 255, len(t))), dtype=np.int32))
    C = C[:, 0:3]

    for i in range(N):
        for j in range(N):
            if D[i, j] <= thresh:
                Y = np.zeros((len(t), 2))
                Y[:, 0] = X[i, 0] + t*(X[j, 0] - X[i, 0])
                Y[:, 1] = X[i, 1] + t*(X[j, 1] - X[i, 1])
                drawLineColored(Y, C)
                #ax.arrow(X[i, 0], X[i, 1], X[j, 0] - X[i, 0], X[j, 1] - X[i, 1], fc="k", ec="k", head_width=0.2, head_length=0.5)
    #Plot cocycle
    for k in range(cocycle.shape[0]):
        [i, j, val] = cocycle[k, :]
        [i, j] = [min(i, j), max(i, j)]
        #plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'r', lineWidth = 3, linestyle='--')
        #ax.arrow(X[i, 0], X[i, 1], X[j, 0] - X[i, 0], X[j, 1] - X[i, 1], fc="k", ec="k", head_width=0.05, head_length=0.1)
        a = 0.5*(X[i, :] + X[j, :])
        plt.text(a[0], a[1], '%g'%val)

    #Enumerate Triangles
    if drawTriangles:
        N = X.shape[0]
        [A, B, C] = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
        [A, B, C] = [A.flatten(), B.flatten(), C.flatten()]
        tidx = np.arange(len(A), dtype=np.int32)
        tidx = tidx[(D[A, B] <= thresh)*(D[B, C] <= thresh)*(D[A, C] <= thresh)]
        [A, B, C] = [A[tidx], B[tidx], C[tidx]]
        plotTriangles(X, A, B, C)

    #Plot X
    plt.scatter(X[:, 0], X[:, 1], 100, 'k')
    for i in range(X.shape[0]):
        plt.text(X[i, 0]+0.02, X[i, 1]+0.02, '%i'%i)

if __name__ == '__main__':
    p = 41
    np.random.seed(10)
    N = 5
    X = np.zeros((N, 2))
    t = np.linspace(0, 1, N+1)[0:N]**1.2
    t = 2*np.pi*t
    X[0:N, 0] = np.cos(t)
    X[0:N, 1] = np.sin(t)
    PDs = doRipsFiltration(X, 1, thresh = -1)
    I = PDs[1]
    thresh = np.max(I[:, 1]) - 0.01

    doRipsFiltrationDionysus(X, coeff = p, thresh = thresh)
