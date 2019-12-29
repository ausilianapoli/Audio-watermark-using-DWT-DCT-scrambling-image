from utils import *
from PIL import Image
from audio_managing import frameToAudio, audioToFrame
from image_managing import binarization, grayscale, imgSize
import numpy as np
import math
import sys

ALPHA = 0.1

#Check if the image is in grayscale and covert it in this mode
def isImgGrayScale(image):
    if image.mode != "L":
        image = grayscale(image)
    return image

#Check if the image is in binary and covert it in this mode
def isImgBinary(image):
    if image.mode != "1":
        image = binarization(image)
    return image

#Embedding of width and heigth. Audio must be linear and not frames
def sizeEmbedding(audio, width, height):
    embedded = audio.copy()

    #Embedding width and heigth
    embedded[0][-1] = width
    embedded[1][-1] = height

    return embedded
    
def sizeExtraction(audio):
    #Extraction of width and heigth
    return int(audio[0][-1]), int(audio[1][-1])
    
#Check if audio is divided in frames:
    # if true, it joins audio and then the inverse will be called
    # if false, it does nothing
def isJoinedAudio(audio):
    if type(audio[0]) in (np.int16, np.int64, np.float64, int, float):
        numOfFrames = -1 #Audio is not divided in frame  
        joinAudio = audio.copy()
    else:
        numOfFrames = audio.shape[0]
        joinAudio = frameToAudio(audio)
    return joinAudio, numOfFrames

def iisJoinedAudio(audio, numOfFrames):
    if type(audio[0]) in (np.int16, np.int64, np.float64, int, float):
        return audioToFrame(audio, numOfFrames)

def LSB(audio, image):   
    image = isImgBinary(image)  
    joinAudio, numOfFrames = isJoinedAudio(audio)
    width, heigth = imgSize(image)

    audioLen = len(joinAudio)
    
    if (width * heigth) + 32 >= audioLen:
        sys.exit("LEAST SIGNIFICANT BIT: Cover dimension is not sufficient for this payload size!")

    joinAudio = sizeEmbedding(joinAudio, width, heigth)

    #Embedding watermark
    for i in range(width):
        for j in range(heigth):
            value = image.getpixel(xy=(i,j))
            value = 1 if value == 255 else 0
            x = i*heigth + j
            joinAudio[x + 32] = setLastBit(joinAudio[x + 32],value)

    if numOfFrames is not -1:
        return audioToFrame(joinAudio, numOfFrames)
    else:
        return joinAudio
    

def iLSB(audio):
    #Verify if audio is divided in frames
    joinAudio, numOfFrames = isJoinedAudio(audio)
    width, heigth = (128,128)#sizeExtraction(joinAudio)
    image = Image.new("1",(width,heigth))

    #Extraction watermark
    for i in range(width):
        for j in range(heigth):
            x = i*heigth + j
            value = getLastBit(joinAudio[x+32])
            image.putpixel(xy=(i,j),value=value)

    return image

#Delta embedding mixed with LSB technique for embedding of width and heigth
def bruteBinary(coeffs, image):
    image = isImgBinary(image)
 
    joinCoeffs = coeffs.copy()

    coeffsLen = len(coeffs)
    frameLen = len(coeffs[0])
    width, heigth = imgSize(image)
    if (width * heigth) + 2 >= coeffsLen:
        sys.exit("DELTA DCT: Cover dimension is not sufficient for this payload size!")

    joinCoeffs = sizeEmbedding(joinCoeffs, width, heigth)

    #Embedding watermark
    for i in range(width):
        for j in range(heigth):
            value = image.getpixel(xy=(i,j))
            x = i*heigth + j
            joinCoeffs[x+2] = setBinary(joinCoeffs[x+2], value)
         
    return joinCoeffs


def ibruteBinary(coeffs):
    joinCoeffs = coeffs.copy()
    width, heigth = (128,128)#sizeExtraction(joinCoeffs)
    extracted = Image.new("L",(width,heigth))
    coeffsLen = len(coeffs)

    #Extraction watermark
    for i in range(width):
        for j in range(heigth):
            x = i*heigth + j
            try:
                value = getBinary(joinCoeffs[x+2])
            except IndexError:
                value = 0
            extracted.putpixel(xy=(i,j),value=value)

    return extracted

#Delta embedding mixed with LSB technique for embedding of width and heigth
def deltaDCT(coeffs, image):
    image = isImgBinary(image)
    width, heigth = imgSize(image)
    
    joinCoeffs = coeffs.copy()

    coeffsLen = len(coeffs)

    #Embedding watermark
    for i in range(width):
        for j in range(heigth):
            value = image.getpixel(xy=(i,j))
            x = i*heigth + j
            v1, v2 = subVectors(joinCoeffs[x])

            norm1, u1 = normCalc(v1)
            norm2, u2 = normCalc(v2)
            
            norm = (norm1 + norm2) / 2
            norm1, norm2 = setDelta(norm, 10, value)

            v1 = inormCalc(norm1, u1)
            v2 = inormCalc(norm2, u2)

            joinCoeffs[x] = isubVectors(v1, v2)
         
    return joinCoeffs


def ideltaDCT(coeffs):
    joinCoeffs = coeffs.copy()
    width, heigth = (128,128)#sizeExtraction(joinCoeffs)
    extracted = Image.new("1",(width,heigth))
    coeffsLen = len(coeffs)

    #Extraction watermark
    for i in range(width):
        for j in range(heigth):
            x = i*heigth + j
            try:
                v1, v2 = subVectors(joinCoeffs[x])
            except IndexError:
                zero = np.zeros(len(joinCoeffs[0]))
                v1, v2 = subVectors(zero)
            norm1, u1 = normCalc(v1)
            norm2, u2 = normCalc(v2)
            
            value = getDelta(norm1 , norm2)
            
            extracted.putpixel(xy=(i,j),value=value)

    return extracted

#Delta embedding mixed with LSB technique for embedding of width and heigth
def bruteGray(coeffs, image):
    image = isImgGrayScale(image)
    
    joinCoeffs = coeffs.copy()

    coeffsLen = len(coeffs)
    frameLen = len(coeffs[0])
    width, heigth = imgSize(image)
    if (width * heigth) + 2 >= coeffsLen:
        sys.exit("DELTA DCT: Cover dimension is not sufficient for this payload size!")

    joinCoeffs = sizeEmbedding(joinCoeffs, width, heigth)

    #Embedding watermark
    for i in range(width):
        for j in range(heigth):
            value = image.getpixel(xy=(i,j))
            x = i*heigth + j
            joinCoeffs[x+2] = setGray(joinCoeffs[x+2], value)
         
    return joinCoeffs


def ibruteGray(coeffs):
    joinCoeffs = coeffs.copy()
    width, heigth = (128,128)#sizeExtraction(joinCoeffs)
    extracted = Image.new("L",(width,heigth))
    coeffsLen = len(coeffs)

    #Extraction watermark
    for i in range(width):
        for j in range(heigth):
            x = i*heigth + j
            try:
                value = getGray(joinCoeffs[x+2])
            except IndexError:
                value = 0
            extracted.putpixel(xy=(i,j),value=value)

    return extracted

#The watermark is embedded into k coefficents of greater magnitudo
def magnitudoDCT(coeffs, watermark, alpha):
    watermark = isImgGrayScale(watermark)
    #print(np.asarray(watermark))
    watermark = createImgArrayToEmbed(watermark)
    #print(watermark)
    coeffs, joinFlag = isJoinedAudio(coeffs)
    if(coeffs.shape[0] < len(watermark)):
        sys.exit("MAGNITUDO DCT: Cover dimension is not sufficient for this payload size!")
    #coeffs = coeffs[:len(watermark)] #to delete for main.py
    wCoeffs = []
    for i in range(len(watermark)):
        wCoeffs.append((coeffs[i])*(1 + alpha*watermark[i]))
    for i in range(len(watermark), len(coeffs)):
        wCoeffs.append(coeffs[i])
    if joinFlag != -1:
        wCoeffs = np.asarray(wCoeffs)
        wCoeffs = iisJoinedAudio(wCoeffs, joinFlag)
    return wCoeffs

#The extraction of watermark from k coefficents of greater magnitudo       
def imagnitudoDCT(coeffs, wCoeffs, alpha):
    coeffs, joinCoeffsFlag = isJoinedAudio(coeffs)
    wCoeffs, joinWCoeffsFlag = isJoinedAudio(wCoeffs)
    #coeffs = coeffs[:len(wCoeffs)]
    watermark = []
    #print(wCoeffs)
    for i in range(len(wCoeffs)):
        #print("wCoeffs[i]: ", wCoeffs[i], " coeffs[i]: ", coeffs[i])
        #if coeffs[i] == 0.0: coeffs[i] = 1.0
        if math.isinf((wCoeffs[i] - coeffs[i])/(coeffs[i])):
            print("wCoeffs[i]: ", wCoeffs[i], " coeffs[i]: ", coeffs[i], "i: ", i)
            watermark.append(abs(math.floor(wCoeffs[i])))
            continue
        if math.isnan((wCoeffs[i] - coeffs[i])/(coeffs[i])): 
            print("wCoeffs[i]: ", wCoeffs[i], " coeffs[i]: ", coeffs[i], "i: ", i)
            watermark.append(0)
            continue
        #print(math.floor(abs((wCoeffs[i] - coeffs[i])/(coeffs[i]*alpha))))
        watermark.append(math.floor(abs((wCoeffs[i] - coeffs[i])/(coeffs[i]*alpha))))
        #watermark.append(abs(math.ceil(wCoeffs[i] - coeffs[i])))
    return convertToPIL(createImgMatrix(extractImage(watermark)))

#Extract image coefficients from global watermark array
def extractImage(watermark):
    nPixel = (watermark[0]*watermark[1])+2
    print(watermark[0], watermark[1])
    print(nPixel)
    return watermark[:nPixel]

#The image becomes matrix from array
def createImgMatrix(image):
    width = image[0]
    heigth = image[1]
    matrixImg = np.reshape(image[2:], (width, heigth))
    print(matrixImg)
    return matrixImg

#Convert numpy type to Image type
def convertToPIL(image):
    PImage = Image.fromarray((image).astype("uint8"), mode="L")
    return PImage

#Routine procedure to embedd the shape of image into flatted array of it
def createImgArrayToEmbed(image):
    width, heigth = imgSize(image)
    flattedImage = [width, heigth]
    tmp = np.ravel(image)
    for i in range(len(tmp)):
        flattedImage.append(tmp[i])
    return flattedImage

'''
TESTING
'''
if __name__ == "__main__":
    
    audio = [1,5,6,7,8,9,4,5,6,1,3,5,4,7,1,5,6,7,8,9,4,5,6,1,3,5,4,7,1,5,6,7,8,9,4,5,6,1,3,5,4,7,5,6,7]
    image = Image.new("1",(3,4))
    image.putpixel(xy=(1,2),value=1)
    lsb = LSB(audio,image)
    print(np.asarray(iLSB(lsb)))
    image = Image.new("L",(3,4))
    image.putpixel(xy=(1,2),value=255)
    image.putpixel(xy=(2,2),value=100)
    image.putpixel(xy=(1,0),value=150)
    image.putpixel(xy=(2,3),value=55)
    image.putpixel(xy=(0,0),value=70)
    delta = deltaDCT(audio, image)
    print(np.asarray(ideltaDCT(audio, delta)))
    
    #flattedImage = createImgArrayToEmbed(image)
    #print("flatted image: ", flattedImage)
    #lenFlattedImage = len(flattedImage)
    #coeffs = audio[:lenFlattedImage]
    wCoeffs = magnitudoDCT(audio, image, ALPHA)
    print("watermarked coeffs: ", wCoeffs)
    watermark = imagnitudoDCT(audio, wCoeffs, ALPHA)
    print("extracted watermark: ", watermark)
    
