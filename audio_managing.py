from scipy.io import wavfile
import subprocess as sp
import platform
from utils import makeFileName
import os
import numpy as np
import math

AUDIO_PATH = 0
SAMPLERATE = 1
AUDIO_DATA = 2

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
def saveWavFile(path, samplerate, signal):
    path = makeFileName("watermarked", path)
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

'''
TESTING
'''

readWavFile()
tupleAudio = readWavFile("piano.wav")
printMetadata(tupleAudio)
print("Is the audio mono? ", isMono(tupleAudio[AUDIO_DATA])) #false
saveWavFile(tupleAudio[AUDIO_PATH], tupleAudio[SAMPLERATE], tupleAudio[AUDIO_DATA])
tupleAudio = joinAudioChannels(tupleAudio[AUDIO_PATH])
printMetadata(tupleAudio)
print("Is the audio mono? ", isMono(tupleAudio[AUDIO_DATA])) #true
frames = toFrame(tupleAudio[AUDIO_DATA],len=1000)
print("Number of frames:", frames.shape) #303 ca



