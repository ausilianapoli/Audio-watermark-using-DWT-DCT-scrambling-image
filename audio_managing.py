from scipy.io import wavfile
import subprocess as sp
import platform
from utils import makeFileName
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pywt

AUDIO_PATH = 0
SAMPLERATE = 1
AUDIO_DATA = 2
WAVELETS_LEVEL = 2

#Read the file audio.wav from path
def readWavFile(path = ""):
    if path == "":
        print("READ WAV FILE must have valid path!")
        return 1
    samplerate, data = wavfile.read(path)
    tupleWav = (path, samplerate, data)
    return tupleWav

#Print some information about file audio    
def printMetadata(entry):
    print("Path: {}"\
          .format(entry[AUDIO_PATH]))
    print("\tsamplerate: {}"\
          .format(entry[SAMPLERATE]))
    print("\t#samples: {}"\
          .format(entry[AUDIO_DATA].shape))

#Check the number of channels of audio file
def isMono(dataAudio):
    return (True if len(dataAudio.shape) == 1 else False)

#Save processed file audio with wav format
def saveWavFile(path, samplerate, signal, prefix):
    path = makeFileName(prefix, path)
    wavfile.write(path, samplerate, signal)
    
#Join audio channels to only one
def joinAudioChannels(path):
    outPath = makeFileName("mono", path)
    if platform.system() == "Linux":
        cmdffmpeg_L = "ffmpeg -y -i {} -ac 1 -f wav {}"\
                    .format(path, outPath)
        os.system(cmdffmpeg_L)
    elif platform.system() == "Windows":
        cmdffmpeg_W = "./ffmpeg/bin/ffmpeg.exe -y -i {} -ac 1 -f wav {}"\
                    .format(path, outPath)
        sp.call(cmdffmpeg_W)
    tupleMono = readWavFile(outPath)
    return tupleMono

#Divide audio in frames
def toFrame(audio, len):
    numFrames = math.ceil(audio.shape[0]/len)
    frames = list()
    for i in range(numFrames):
        frames.append(audio[i*len : (i*len)+len])
    
    return np.asarray(frames)

#Plot the waveform of input audio file
def waveform(entry):
    plt.figure()
    plt.plot(entry[AUDIO_DATA])
    plt.title("Waveform: {}"\
              .format(entry[AUDIO_PATH]))
    plt.show()
    
#Get the list of all wavelets
def getWaveletsFamilies():
    return pywt.families(short = True)

#Get the list of all signal extension modes
def getWaveletsModes():
    return pywt.Modes.modes

#Multilevel DWT
def DWT(data, wavelet, mode, level):
    coeffs = pywt.wavedec(data, wavelet, mode, level)
    #cA2, cD2, cD1 = coeffs
    return coeffs

def iDWT(coeffs, wavelet, mode):
    data = pywt.waverec(coeffs, wavelet, mode)
    return data

'''
TESTING
'''

readWavFile()
tupleAudio = readWavFile("piano.wav")
printMetadata(tupleAudio)
print("Is the audio mono? ", isMono(tupleAudio[AUDIO_DATA])) #false
saveWavFile(tupleAudio[AUDIO_PATH], tupleAudio[SAMPLERATE], tupleAudio[AUDIO_DATA], "watermarked")
tupleAudio = joinAudioChannels(tupleAudio[AUDIO_PATH])
printMetadata(tupleAudio)
print("Is the audio mono? ", isMono(tupleAudio[AUDIO_DATA])) #true
frames = toFrame(tupleAudio[AUDIO_DATA],len=1000)
print("Number of frames:", frames.shape) #303 ca
waveform(tupleAudio)
waveletsFamilies = getWaveletsFamilies()
waveletsModes = getWaveletsModes()
coeffs = DWT(tupleAudio[AUDIO_DATA], waveletsFamilies[0], waveletsModes[0], WAVELETS_LEVEL)
print("wavelets coeffs: ", coeffs)
cA2, cD2, cD1 = coeffs
print("cA2: ", cA2, "\ncD2: ", cD2, "\ncD1: ", cD1)
#cA2 = abs(cA2)
#cD2 = abs(cD2)
#scD1 = abs(cD1)
coeffs = cA2, cD2, cD1
data = iDWT(coeffs, waveletsFamilies[0], waveletsModes[0])
print("iDWT data: ", data)
data = np.int16(data)
print("iDWT == data audio? ", data == tupleAudio[AUDIO_DATA])
saveWavFile(tupleAudio[AUDIO_PATH], tupleAudio[SAMPLERATE], data, "dwt")



