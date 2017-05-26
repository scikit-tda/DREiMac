"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: To wrap around Ripser to compute persistence diagrams
"""
import subprocess
import os
import numpy as np
import time
import matplotlib.pyplot as plt

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

#Wrapper around Uli Bauer's ripser code
def doRipsFiltration(X, maxHomDim, thresh = -1, coeff = 2):
    #Compute all pairwise distances assuming Euclidean
    XSqr = np.sum(X**2, 1)
    D = XSqr[:, None] + XSqr[None, :] - 2*X.dot(X.T)
    D[D < 0] = 0 #Numerical precision
    D = np.sqrt(D)
    return doRipsFiltrationDM(D, maxHomDim, thresh, coeff)

if __name__ == '__main__':
    np.random.seed(10)
    X = np.random.randn(200, 2)
    X = X/np.sqrt(np.sum(X**2, 1)[:, None])
    #plt.plot(X[:, 0], X[:, 1], '.')
    #plt.show()
    PDs = doRipsFiltration(X, 1, coeff = 3)
    plotDGM(PDs[1])
    plt.show()
