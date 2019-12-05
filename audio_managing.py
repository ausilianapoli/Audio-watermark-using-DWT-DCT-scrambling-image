from scipy.io import wavfile
import numpy as np
import subprocess as sp
import os
import platform

AUDIO_PATH = 0
SAMPLERATE = 1
AUDIO_DATA = 2

#Read the file audio.wav from path
def readWavFile(path):
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

#Save processed file audio with wav format
def saveFile(path, samplerate, signal):
    fileName = os.path.basename(path)
    dirName = os.path.dirname(path)
    fileName = "watermarked_" + fileName
    path = os.path.join(dirName, fileName)
    wavfile.write(path, samplerate, signal)

'''
TESTING
'''

tupleAudio = readWavFile("piano.wav")
printMetadata(tupleAudio)
saveFile(tupleAudio[AUDIO_PATH], tupleAudio[SAMPLERATE], tupleAudio[AUDIO_DATA])


