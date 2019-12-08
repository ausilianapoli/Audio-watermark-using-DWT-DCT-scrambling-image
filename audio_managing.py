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

DWT_SET = set({"haar",
               "bior1.1", "bior1.3", "bior1.5",
               "bior2.2", "bior2.4", "bior2.6", "bior2.8",
               "bior3.1", "bior3.3", "bior3.5", "bior3.7", "bior3.9",
               "bior4.4",
               "bior5.5",
               "bior6.8",
               "coif1", "coif2", "coif3", "coif4", "coif5", "coif6", "coif7", "cdb3oif8", "coif9", "coif10", "coif11", "coif12", "coif13", "coif14", "coif15", "coif16", "coif17",
               "db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9", "db10", "db11", "db12", "db13", "db14", "db15", "db16", "db17", "db18", "db19", "db20", "db21", "db22", "db23", "db24", "db25", "db26", "db27", "db28", "db29", "db30", "db31", "db32", "db33", "db34", "db35", "db36", "db37", "db38",
               "dmey",
               "rbio1.1", "rbio1.3", "rbio1.5",
               "rbio2.2", "rbio2.4", "rbio2.6", "rbio2.8",
               "rbio3.1", "rbio3.3", "rbio3.5", "rbio3.7", "rbio3.9",
               "rbio4.4",
               "rbio5.5",
               "rbio6.8",
               "sym2", "sym3", "sym4", "sym5", "sym6", "sym7", "sym8", "sym9", "sym10", "sym11", "sym12", "sym13", "sym14", "sym15", "sym16", "sym17", "sym18", "sym19", "sym20"})

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
    return pywt.wavelist()

#Get the list of all signal extension modes
def getWaveletsModes():
    return pywt.Modes.modes

def filterWaveletsFamilies(families):
    DWTFamilies = list(filter(lambda w: w in DWT_SET, families))
    return DWTFamilies

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
DWTFamilies = filterWaveletsFamilies(waveletsFamilies)
print("DWT Families: ", DWTFamilies)
print("len DWT Families = ", len(DWTFamilies))
waveletsModes = getWaveletsModes()
coeffs = DWT(tupleAudio[AUDIO_DATA], DWTFamilies[DWTFamilies.index("haar")], waveletsModes[waveletsModes.index("symmetric")], WAVELETS_LEVEL)
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




