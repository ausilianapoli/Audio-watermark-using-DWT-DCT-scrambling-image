from scipy.io import wavfile
import numpy as np
import subprocess as sp
import os
import platform

#Read the file audio.wav from path
def readWavFile(path):
    samplerate, data = wavfile.read(path)
    tupleWav = (path, samplerate, data)
    return tupleWav

#Print some information about file audio    
def printMetadata(entry):
    print("Path: {}"\
          .format(entry[0]))
    print("\tsamplerate: {}"\
          .format(entry[1]))
    print("\t#samples: {}"\
          .format(entry[2].shape))
    ttl = list(entry[2].shape) #to extract tuple's values as int first it converts into list
    shape = ttl.pop() #and then it is popped the single element of list
    print("\tduration: {} seconds"\
          .format(str(round(shape/entry[1]))))

'''
TESTING
'''

tupleAudio = readWavFile("piano.wav")
printMetadata(tupleAudio)


