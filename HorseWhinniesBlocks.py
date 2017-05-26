"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: To show how TDA can be used to quantify quasiperiodicity
in an audio clip of horse whinnies
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.interpolate as interp
from TDA import *
from SlidingWindow import *
import scipy.io.wavfile
import scipy.io as sio


def getPeriodicityScores(I1Z2, I1Z3, I2):
    (Z2H1Max1, Z2H1Max2, Z3H1Max1, Z3H1Max2, Z3H1Max3, H2Max1) = (0, 0, 0, 0, 0, 0)
    idx = np.argsort(I1Z3[:, 0] - I1Z3[:, 1])
    if len(idx) > 0:
        Z3H1Max1 = I1Z3[idx[0], 1] - I1Z3[idx[0], 0]
    if len(idx) > 1:
        Z3H1Max2 = I1Z3[idx[1], 1] - I1Z3[idx[1], 0]
    if len(idx) > 2:
        Z3H1Max3 = I1Z3[idx[2], 1] - I1Z3[idx[2], 0]
    idx = np.argsort(I1Z2[:, 0] - I1Z2[:, 1])
    if len(idx) > 0:
        Z2H1Max1 = I1Z2[idx[0], 1] - I1Z2[idx[0], 0]
    if len(idx) > 1:
        Z2H1Max2 = I1Z2[idx[1], 1] - I1Z2[idx[0], 0]
    idx = np.argsort(I2[:, 0] - I2[:, 1])
    if len(idx) > 0:
        H2Max1 = I2[idx[0], 1] - I2[idx[0], 0]
    #Periodicity Score
    PScore = max(Z3H1Max1/np.sqrt(3), Z2H1Max1/np.sqrt(3))
    #Modified Periodicity Score
    PScoreMod = (Z3H1Max1 - Z3H1Max2)/np.sqrt(3)
    PScoreMod = max((Z2H1Max1 - Z2H1Max2)/np.sqrt(3), PScoreMod)
    #Harmonic Subscore
    HSubscore = 0
    if Z3H1Max1 > 0:
        HSubscore = 1 - Z2H1Max1/Z3H1Max1
    elif Z2H1Max1 > 0:
        HSubscore = 1
    #Quasiperiodicity Score
    QPScore = np.sqrt(Z3H1Max2*H2Max1/3.0)
    
    #H1 only quasiperiodicity score
    QPScoreMod = (Z3H1Max2 - Z3H1Max3)/np.sqrt(3)
    return (PScore, PScoreMod, HSubscore, QPScore, QPScoreMod)

if __name__ == '__main__':
    doPlot = True
    #Read in the audio file.  Fs is the sample rate, and
    #X is the audio signal
    Fs, X = scipy.io.wavfile.read("horsewhinnie.wav")
    
    #These variables are used to adjust the window size
    F0 = 493 #First fundamental frequency
    G0 = 1433 #Second fundamental frequency

    W = int(round(Fs/G0))
    print "W = ", W
    Win = 1024
    hop = 128
    NBlocks = int(np.ceil(1 + (len(X) - Win)/hop))
    
    PScores = np.zeros(NBlocks)
    QPScores = np.zeros(NBlocks)
    QPScoresMod = np.zeros(NBlocks)
    plt.figure(figsize=(12, 6))
    for i in range(NBlocks):
        print "Processing block %i of %i"%(i, NBlocks)
        x = X[i*hop:i*hop+Win]        

        #Step 2: Get the sliding window embedding
        Y = getSlidingWindow(x, W, 2, 2)
        #Mean-center and normalize
        Y = Y - np.mean(Y, 1)[:, None]
        Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]

        #Step 3: Do the 1D and 2D rips filtrations
        PDs = doRipsFiltration(Y, 2)
        I1 = PDs[1]
        I2 = PDs[2]
        (PScore, PScoreMod, HSubscore, QPScore, QPScoreMod) = getPeriodicityScores(I1, I1, I2)
        PScores[i] = PScoreMod
        QPScores[i] = QPScore
        QPScoresMod[i] = QPScoreMod
        sio.savemat("PScores.mat", {"PScores":PScores, "QPScores":QPScores, "QPScoresMod":QPScoresMod})
        
        if doPlot:
            plt.clf()
            plt.subplot(121)
            time = float(i)*hop/Fs
            plt.title("PScore = %.3g\nQPScore = %.3g\nQPScoreMod = %.3g"%(PScores[i], QPScores[i], QPScoresMod[i]))
            plt.plot(time + np.arange(Win)/float(Fs), x)
            plt.ylim([np.min(X), np.max(X)])
            plt.xlabel("Time")
            plt.subplot(122)
            plotDGM(I1, 'r')
            plt.hold(True)
            plotDGM(I2, 'g')
            plt.xlim([0, 2])
            plt.ylim([0, 2])
            plt.savefig("%i.png"%i, bbox_inches = 'tight')
