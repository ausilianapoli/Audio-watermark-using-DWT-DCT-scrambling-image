import subprocess as sp
import platform
from utils import makeFileName
import os

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