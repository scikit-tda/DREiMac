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
import librosa
from sklearn.decomposition import PCA


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

def averageSnippet(x, Win, hop):
    y = np.zeros(Win)
    NBlocks = int(np.floor(1 + (len(x) - Win)/hop))
    for i in range(NBlocks):
        z = x[i*hop:i*hop+Win]
        y += z
    y = y/float(NBlocks)
    return y

def averageSnippetFrequency(x, Win, hop, NIters = 10, doPlot = False, hamming = False):
    """
    Do averaging in the frequency domain and do phase retreival
    to get a time domain signal
    """
    Y = np.zeros(Win)
    NBlocks = int(np.floor(1 + (len(x) - Win)/hop))
    print "NBlocks = ", NBlocks
    Ham = np.ones(Win)
    if not hamming:
        Ham = np.hamming(Win)
    for i in range(NBlocks):
        z = x[i*hop:i*hop+Win]
        Y += np.abs(np.fft.fft(z*Ham))
    Y = Y/float(NBlocks)
    A = np.array(Y)
    #Now run Griffin Lim
    for it in range(NIters):
        print "Iteration %i"%it
        A = np.fft.fft(np.fft.ifft(A))
        norm = np.sqrt(A*np.conj(A))
        norm[norm < 1e-5] = 0
        A = Y*A/norm
    x = np.fft.ifft(A)
    x = np.real(x)
    Y = Y/np.max(Y)
    Y2 = np.abs(np.fft.fft(x))
    Y2 = Y2/np.max(Y2)
    if doPlot:
        plt.plot(Y, 'b')
        plt.plot(Y2, 'r')
        plt.show()
    return (x, Y)

def doSnippetPCA(x, Win, hop):
    y = np.zeros(Win)
    NBlocks = int(np.floor(1 + (len(x) - Win)/hop))
    X = np.zeros((NBlocks, Win))
    for i in range(NBlocks):
        X[i, :] = x[i*hop:i*hop+Win]
    print "X.shape = ", X.shape
    pca = PCA(n_components = 20)
    Y = pca.fit_transform(X)
    A = pca.components_
    eigs = pca.explained_variance_ratio_
    print "Explained Variance: ", np.sum(eigs)
    print "A.shape = ", A.shape
    return A


if __name__ == '__main__':
    Fs, X = scipy.io.wavfile.read("horsewhinnie.wav")
    x = X[int(Fs*0.87):int(Fs*1)]

    #These variables are used to adjust the window size
    F0 = 540 #493 #First fundamental frequency
    G0 = 1433 #Second fundamental frequency

    W = int(round(Fs/F0))
    print "W = ", W
    Win = 512
    freqs = (np.arange(Win)/float(Win))*Fs
    NHarmonics = 8

    PCs = doSnippetPCA(x, Win, 1)

    for i in range(PCs.shape[0]):
        x = PCs[i, :]

        #Do FFT
        F = np.abs(np.fft.fft(x))
        F = np.log(F)

        #Step 2: Get the sliding window embedding
        Y = getSlidingWindow(x, W, 1, 1)
        #Mean-center and normalize
        Y = Y - np.mean(Y, 1)[:, None]
        Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]
        #Step 3: Do the 1D and 2D rips filtrations
        PDs = doRipsFiltration(Y, 2, coeff = 3)
        I1 = PDs[1]
        I2 = PDs[2]
        (PScore, PScoreMod, HSubscore, QPScore, QPScoreMod) = getPeriodicityScores(I1, I1, I2)


        plt.clf()
        plt.subplot(221)
        plt.title("PScore = %.3g\nQPScore = %.3g\nQPScoreMod = %.3g"%(PScore, QPScore, QPScoreMod))
        #plt.plot(np.arange(len(y))/float(Fs), y)
        plt.plot(x)
        plt.ylim([np.min(x), np.max(x)])
        plt.xlabel("Time")
        plt.subplot(222)
        plotDGM(I1, 'r')
        plt.hold(True)
        plotDGM(I2, 'g')
        plt.xlim([0, 2])
        plt.ylim([0, 2])
        plt.subplot2grid((2, 2), (1, 0), colspan = 20)
        F = np.abs(np.fft.fft(x))
        F = np.log(F)
        plt.plot(freqs, F)
        plt.xlabel("Frequency (Hz)")
        plt.hold(True)
        for k in range(1, NHarmonics):
            plt.plot([F0*k, F0*k], [np.min(F), np.max(F)], 'b')
            plt.plot([G0*k, G0*k], [np.min(F), np.max(F)], 'r')
        plt.xlim([0, 8000])

        plt.savefig("PC%i.svg"%i, bbox_inches = 'tight')



if __name__ == '__main__2':
    doPlot = True
    #Read in the audio file.  Fs is the sample rate, and
    #X is the audio signal
    Fs, X = scipy.io.wavfile.read("horsewhinnie.wav")

    #These variables are used to adjust the window size
    F0 = 540 #493 #First fundamental frequency
    G0 = 1433 #Second fundamental frequency

    W = int(round(Fs/G0))
    print "W = ", W
    Win = 2048
    hop = 128
    NBlocks = int(np.ceil(1 + (len(X) - Win)/hop))
    freqs = (np.arange(Win)/float(Win))*Fs
    NHarmonics = 8

    PScores = np.zeros(NBlocks)
    QPScores = np.zeros(NBlocks)
    QPScoresMod = np.zeros(NBlocks)
    plt.figure(figsize=(12, 12))
    for i in range(200, NBlocks):
        print "Processing block %i of %i"%(i, NBlocks)
        x = X[i*hop:i*hop+Win]

        #Do FFT
        F = np.abs(np.fft.fft(x))
        F = np.log(F)

        #Step 2: Get the sliding window embedding
        Y = getSlidingWindow(x, W, 2, 4)
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
            plt.subplot(221)
            time = float(i)*hop/Fs
            plt.title("PScore = %.3g\nQPScore = %.3g\nQPScoreMod = %.3g"%(PScores[i], QPScores[i], QPScoresMod[i]))
            plt.plot(time + np.arange(Win)/float(Fs), x)
            plt.ylim([np.min(X), np.max(X)])
            plt.xlabel("Time")
            plt.subplot(222)
            plotDGM(I1, 'r')
            plt.hold(True)
            plotDGM(I2, 'g')
            plt.xlim([0, 2])
            plt.ylim([0, 2])
            plt.subplot2grid((2, 2), (1, 0), colspan = 20)
            plt.plot(freqs[0:Win/2], F[0:Win/2])
            plt.xlabel("Frequency (Hz)")
            plt.hold(True)
            for k in range(1, NHarmonics):
                plt.plot([F0*k, F0*k], [0, np.max(F)], 'b')
                plt.plot([G0*k, G0*k], [0, np.max(F)], 'r')
            plt.xlim([0, 8000])

            plt.savefig("%i.png"%i, bbox_inches = 'tight')
