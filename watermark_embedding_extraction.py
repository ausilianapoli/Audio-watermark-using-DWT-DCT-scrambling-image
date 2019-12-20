from utils import setLastBit, getLastBit, decToBinary, binaryToDec, normalize, inormalize
from PIL import Image
from audio_managing import frameToAudio
from image_managing import binarization, grayscale, imgSize
import numpy as np
import math

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
def sizeEmbedding(audio, width, heigth):
    bWidth = decToBinary(width, 16)
    bheigth = decToBinary(heigth, 16)

    embedded = audio.copy()

    #Embedding width and heigth
    for w in range(16):
        embedded[w] = setLastBit(embedded[w],int(bWidth[w]))
        embedded[w+16] = setLastBit(embedded[w+16],int(bheigth[w]))

    return embedded

def sizeExtraction(audio):
    bWidth, bheigth = ("","")

    #Extraction of width and heigth
    for w in range(16):
        bWidth += str(getLastBit(audio[w]))
        bheigth += str(getLastBit(audio[w+16]))

    width = binaryToDec(bWidth)
    heigth = binaryToDec(bheigth)

    return width, heigth

#Check if audio is divided in frames:
    # if true, it joins audio and then the inverse will be called
    # if false, it does nothing
def isJoinedAudio(audio):
    if type(audio[0]) in (np.int16, np.float64, int, float):
        numOfFrames = -1 #Audio is not divided in frame  
        joinAudio = audio.copy()
    else:
        numOfFrames = audio.shape[0]
        joinAudio = frameToAudio(audio)
    return joinAudio, numOfFrames

def iisJoinedAudio(audio):
    return audioToFrame(joinAudio, numOfFrames)

def LSB(audio, image):   
    image = isImgBinary(image)  
    joinAudio, numOfFrames = isJoinedAudio(audio)
    width, heigth = imgSize(image)

    audioLen = len(joinAudio)
    
    if (width * heigth) + 32 >= audioLen:
        print("LEAST SIGNIFICANT BIT: Cover dimension is not sufficient for this payload size!")
        return

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
    width, heigth = sizeExtraction(joinAudio)
    image = Image.new("1",(width,heigth))

    #Extraction watermark
    for i in range(width):
        for j in range(heigth):
            x = i*heigth + j
            value = getLastBit(joinAudio[x+32])
            image.putpixel(xy=(i,j),value=value)

    return image

#Delta embedding mixed with LSB technique for embedding of width and heigth
def deltaDCT(coeffs, image):
    image = isImgGrayScale(image)
    joinCoeffs, numOfFrames = isJoinedAudio(coeffs)
    coeffsLen = len(joinCoeffs)
    width, heigth = imgSize(image)
    if (width * heigth) + 32 >= coeffsLen:
        print("DELTA DCT: Cover dimension is not sufficient for this payload size!")
        return

    joinCoeffs = sizeEmbedding(joinCoeffs, width, heigth)

    #Embedding watermark
    for i in range(width):
        for j in range(heigth):
            value = image.getpixel(xy=(i,j))
            x = i*heigth + j
            joinCoeffs[x+32] = joinCoeffs[x+32] + normalize(value,255)
            
    if numOfFrames is not -1:
        return audioToFrame(joinCoeffs, numOfFrames)
    else:
        return joinCoeffs

def ideltaDCT(coeffs, wCoeffs):
    joinCoeffs, _ = isJoinedAudio(coeffs)
    joinWCoeffs, _ = isJoinedAudio(wCoeffs)
    
    width, heigth = sizeExtraction(joinWCoeffs)
    extracted = Image.new("L",(width,heigth))
    coeffsLen = len(coeffs)

    #Extraction watermark
    for i in range(width):
        for j in range(heigth):
            x = i*heigth + j
            value = inormalize(abs(joinWCoeffs[x+32] - joinCoeffs[x+32]), 255)
            extracted.putpixel(xy=(i,j),value=value)

    return extracted
    
#The watermark is embedded into k coefficents of greater magnitudo
def magnitudoDCT(coeffs, watermark, alpha):
    watermark = isImgGrayScale(watermark)
    #print(np.asarray(watermark))
    watermark = createImgArrayToEmbed(watermark)
    #print(watermark)
    coeffs, joinFlag = isJoinedAudio(coeffs)
    #coeffs = coeffs[:len(watermark)] #to delete for main.py
    wCoeffs = []
    for i in range(len(watermark)):
        wCoeffs.append((coeffs[i])*(1 + alpha*watermark[i]))
    for i in range(len(watermark), len(coeffs)):
        wCoeffs.append(coeffs[i])
    if joinFlag != -1:
        wCoeffs = iisJoinedAudio(wCoeffs)
    return wCoeffs

#The extraction of watermark from k coefficents of greater magnitudo       
def imagnitudoDCT(coeffs, wCoeffs, alpha):
    coeffs, joinCoeffsFlag = isJoinedAudio(coeffs)
    wCoeffs, joinWCoeffsFlag = isJoinedAudio(wCoeffs)
    #coeffs = coeffs[:len(wCoeffs)]
    watermark = []
    for i in range(len(wCoeffs)):
        watermark.append(math.floor((wCoeffs[i] - coeffs[i])/(coeffs[i]*alpha)))
        #watermark.append(math.ceil(wCoeffs[i] - coeffs[i]))
    return convertToPIL(createImgMatrix(extractImage(watermark)))

def extractImage(watermark):
    nPixel = (watermark[0]*watermark[1])+2
    return watermark[:nPixel]

def createImgMatrix(image):
    width = image[0]
    heigth = image[1]
    matrixImg = np.reshape(image[2:], (width, heigth))
    return matrixImg

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
    
