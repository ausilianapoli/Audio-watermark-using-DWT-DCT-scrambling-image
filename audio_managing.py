from scipy.io import wavfile
import numpy as np
import subprocess as sp
import os
import platform

#Read the file audio.wav from path
def readWavFile(self, path):
    samplerate, data = wavfile.read(path)
    tupleWav = (path, samplerate, data)
    return tupleWav


'''
TESTING
'''

tupleAudio = readWavFile("piano.wav")

