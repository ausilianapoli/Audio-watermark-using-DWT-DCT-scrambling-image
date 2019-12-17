from utils import setLastBit, getLastBit, decToBinary, binaryToDec
from PIL import Image
from audio_managing import frameToAudio
import numpy as np
import math

ALPHA = 0.001

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
            x = i*height + j
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

    #Extraction watermark
    for i in range(width):
        for j in range(height):
            x = i*height + j
            value = getLastBit(joinAudio[x+32])
            image.putpixel(xy=(i,j),value=value)

    return image
    
def magnitudoDCT(coeffs, watermark, alpha):
    wCoeffs = []
    if(len(coeffs) == len(watermark)):
        for i in range(len(coeffs)):
            wCoeffs.append(((coeffs[i])*(1 + alpha*watermark[i])))
        return wCoeffs
    else:
        print("magnitudoDCT: error because DCT coefficients and watermark coefficients must have same length")
        return None
    
        
def imagnitudoDCT(coeffs, wCoeffs, alpha):
    watermark = []
    for i in range(len(coeffs)):
        watermark.append(math.ceil((wCoeffs[i] - coeffs[i])/(coeffs[i]*alpha)))
    return watermark

def createImgArrayToEmbed(image):
    width, heigth = image.shape
    flattedImage = [width, heigth]
    tmp = np.ravel(image)
    for i in range(len(tmp)):
        flattedImage.append(tmp[i])
    return flattedImage

audio = [1,5,6,7,8,9,4,5,6,1,3,5,4,7,1,5,6,7,8,9,4,5,6,1,3,5,4,7,1,5,6,7,8,9,4,5,6,1,3,5,4,7,5,6,7]
image = Image.new("1",(3,4))
image.putpixel(xy=(1,2),value=1)
lsb = LSB(audio,image)
print(np.asarray(iLSB(lsb)))

flattedImage = createImgArrayToEmbed(image)
print(flattedImage)
wCoeffs = magnitudoDCT(audio[:11], flattedImage, ALPHA)
print(wCoeffs)
watermark = imagnitudoDCT(audio[:11], wCoeffs, ALPHA)
print(watermark)

