from utils import setLastBit, numberToBinary
from PIL import Image
from audio_managing import frameToAudio
import numpy as np

def leastSignificantBit(audio, image):
    if image.mode is not "1":
        print("LEAST SIGNIFICANT BIT: Image must be binary!")
        return
    
    #Verify if audio is divided in frames
    if type(audio[0][0]) is not int:
        numOfFrames = audio.shape[0]
        joinAudio = frameToAudio(audio)
        frames = 1
    else:
        numOfFrames = -1 #Audio is not divided in frame  

    width, heigth = image.size
    audioLen = joinAudio.shape[0]
    
    if (width * heigth) + 32 >= audioLen:
        print("LEAST SIGNIFICANT BIT: Cover dimension is not sufficient for this payload size!")
        return

    bWidth = numberToBinary(width, 16)
    bHeigth = numberToBinary(heigth, 16)

    #Embedding width and heigth
    for w in range(16):
        joinAudio[w] = setLastBit(joinAudio[w],bWidth[w])
        joinAudio[w+16] = setLastBit(joinAudio[w+16],bHeigth[w])

    #Embedding watermark
    for i in range(width):
        for j in range(heigth):
            value = image.getpixel(xy=(i,j))
            x = i*width + j
            joinAudio[x + 32] = setLastBit(joinAudio[x + 32],value)

    if numOfFrames is not -1:
        return audioToFrame(joinAudio)
    else:
        return joinAudio
    