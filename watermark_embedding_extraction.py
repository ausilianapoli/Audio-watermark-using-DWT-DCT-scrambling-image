from utils import setLastBit, getLastBit, decToBinary, binaryToDec
from PIL import Image
from audio_managing import frameToAudio
import numpy as np

def LSB(audio, image):
    
    if image.mode is not "1":
        print("LEAST SIGNIFICANT BIT: Image must be binary!")
        return
    
    #Verify if audio is divided in frames
    if type(audio[0]) is int:
        numOfFrames = -1 #Audio is not divided in frame  
        joinAudio = audio
    else:
        numOfFrames = audio.shape[0]
        joinAudio = frameToAudio(audio)

    width, height = image.size
    audioLen = len(joinAudio)
    
    if (width * height) + 32 >= audioLen:
        print("LEAST SIGNIFICANT BIT: Cover dimension is not sufficient for this payload size!")
        return

    bWidth = decToBinary(width, 16)
    bHeight = decToBinary(height, 16)

    #Embedding width and heigth
    for w in range(16):
        joinAudio[w] = setLastBit(joinAudio[w],int(bWidth[w]))
        joinAudio[w+16] = setLastBit(joinAudio[w+16],int(bHeight[w]))

    #Embedding watermark
    for i in range(width):
        for j in range(height):
            value = image.getpixel(xy=(i,j))
            x = i*width + j
            joinAudio[x + 32] = setLastBit(joinAudio[x + 32],value)

    if numOfFrames is not -1:
        return audioToFrame(joinAudio, numOfFrames)
    else:
        return joinAudio
    

def iLSB(audio):
    #Verify if audio is divided in frames
    if type(audio[0]) is int:
        numOfFrames = -1 #Audio is not divided in frame  
        joinAudio = audio
    else:
        numOfFrames = audio.shape[0]
        joinAudio = frameToAudio(audio)
    
    bWidth, bHeight = ("","")

    #Extraction of width and height
    for w in range(16):
        bWidth += str(getLastBit(joinAudio[w]))
        bHeight += str(getLastBit(joinAudio[w+16]))

    width = binaryToDec(bWidth)
    height = binaryToDec(bHeight)
    
    image = Image.new("1",(width,height))

    #Embedding watermark
    for i in range(width):
        for j in range(height):
            x = i*width + j
            value = getLastBit(joinAudio[x+32])
            image.putpixel(xy=(i,j),value=value)

    return image
    
