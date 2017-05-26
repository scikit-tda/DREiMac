"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: To show how TDA can be used to quantify quasiperiodicity
in an audio clip of horse whinnies
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.interpolate as interp
import scipy.signal

from TDA import *
from SlidingWindow import *
import scipy.io.wavfile
import scipy.io as sio

def getTimeDerivative(I, Win):
    dw = np.floor(Win/2)
    t = np.arange(-dw, dw+1)
    sigma = 0.4*dw
    xgaussf = t*np.exp(-t**2  / (2*sigma**2))
    #Normalize by L1 norm to control for length of window
    xgaussf = xgaussf/np.sum(np.abs(xgaussf))
    IRet = scipy.signal.convolve(I, xgaussf, 'valid')
    return IRet

if __name__ == '__main__':
    #Read in the audio file.  Fs is the sample rate, and
    #X is the audio signal
    Fs, X = scipy.io.wavfile.read("horsewhinniestereo.wav")
    
    #These variables are used to adjust the window size
    F0 = 493 #First fundamental frequency
    G0 = 1433 #Second fundamental frequency

    ###TODO: Modify this variable (time in seconds)
    time = 0.84

    #Step 1: Extract an audio snippet starting at the chosen time
    SigLen = 1024 #The number of samples to take after the start time
    Tau = 2
    dT = 2
    iStart = int(round(time*Fs))
    x = X[iStart:iStart + SigLen]
    #x = getTimeDerivative(x, 30)
    W = int(round(1.5*Fs/G0))

    #Step 2: Get the sliding window embedding
    Y = getSlidingWindow(x, W*2, Tau, dT)
    #Mean-center and normalize
    Y = Y - np.mean(Y, 1)[:, None]
    Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]
    sio.savemat("Y.mat", {"Y":Y})

    #Step 3: Do the 1D rips filtration
    PDs = doRipsFiltration(Y, 2)
    PD = PDs[1]
    I1 = PDs[1]
    I2 = PDs[2]

    #Step 4: Figure out the second largest persistence
    sP = 0
    sPIdx = 0
    if PD.shape[0] > 1:
        Pers = PD[:, 1] - PD[:, 0]
        sPIdx = np.argsort(-Pers)[1]
        sP = Pers[sPIdx]
        
    #Step 5: Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("Starting At %g Seconds"%time)
    plt.plot(time + np.arange(len(x))/float(Fs), x)
    plt.xlabel("Time")
    plt.subplot(122)
    plotDGM(I1, 'r')
    plt.hold(True)
    plotDGM(I2, 'g')
    plt.hold(True)
    #plt.plot([PD[sPIdx, 0]]*2, PD[sPIdx, :], 'r')
    plt.scatter(PD[sPIdx, 0], PD[sPIdx, 1], 20, 'r')
    plt.title("Second Largest Persistence: %g"%sP)
    
    plt.show()
