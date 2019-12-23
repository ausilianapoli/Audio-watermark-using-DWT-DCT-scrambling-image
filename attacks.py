import subprocess as sp
import platform
import os
import numpy as np
from utils import makeFileName

#The audio amplitude is scaling of factor t
def amplitudeScaling(data, t):
    return data*t

#The audio is resampled
def resampling(path, sampleRate):
    name = makeFileName(str(sampleRate), path)
    if platform.system() == "Linux":
        cmdffmpegL = "ffmpeg -y -i {} -ar {} -f wav {}"\
                    .format(path, sampleRate, name)
        os.system(cmdffmpegL)
    elif platform.system() == "Windows":
        cmdffmpegW = "./ffmpeg/bin/ffmpeg -y -i {} -ar {} -f wav {}"\
                    .format(path, sampleRate, name)
        sp.call(cmdffmpegW)
    return name

#It calculates the Low Pass Butterworth based on its mathematic formula (it's used in frequency domain)
def butterLPFilter(data, frequency, n = 1): #n is order filter
    mask = np.zeros(data.size) #it will be my filter
    for i in range(int(len(mask)/2)):
        mask[i] = 1/(1 + (i/frequency)**(2*n))
        mask[len(mask) - 1 - i] =  mask[i]
    return mask*data