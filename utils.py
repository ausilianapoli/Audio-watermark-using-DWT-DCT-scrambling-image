import os
import math
import numpy as np
import bitstring
import sys
from PIL import Image
import image_managing as im

#Return inverse module m of a
def imodule(a, m):
    a = a % m
    for x in range(m):
        if (a * x) % m == 1:
            return x   
    return 1

#Return first or second number coprime with m
def coprime(m, mode="first"):
    if mode is not "first" and mode is not "second":
        sys.exit("COPRIME: Mode must be first or second!")

    found = 0
    for x in range(2,m):
        if math.gcd(x,m) == 1:
            if mode is "first":
                return x
            elif mode is "second":
                if found < 1:
                    found += 1
                else:
                    return x

    return 1

#Routine to create path for file to save
def makeFileName(prefix, path):
    fileName = os.path.basename(path)
    dirName = os.path.dirname(path)
    fileName = prefix + "-" + fileName
    nPath = os.path.join(dirName, fileName)
    return nPath

def setLastBit(number, bit):
    if type(number) in (int, np.int16):
        return ((number >> 1) << 1) | bit
    elif type(number) in (float, np.float64):
        binary = decToBinary(number)
        binary = binary[:23] + str(bit) + binary[24:]
        return binaryToDec(binary)

def getLastBit(number):
    if type(number) in (int, np.int16):
        return int(number % 2)
    elif type(number) in (float, np.float64):
        binary = decToBinary(number)
        return int(binary[23])

#Return a string containing the number written in binary notation with bits bit
def decToBinary(number, bits=16):
    if type(number) in (int, np.int16):
        if number > 2**bits:
            sys.exit("DEC TO BINARY: Insufficient number of bit!")

        binary = ""
        for i in range(bits):
            if number != 0:
                bit = number % 2
                number = number >> 1
                binary += str(bit)
            else:
                binary += str(0)
        
        return binary[::-1]

    elif type(number) in (float, np.float64):
        b = bitstring.BitArray(float=number, length=32).bin
        return b[:9] + " " + b[9:]

#Return a int from a string containing the number written in binary notation
def binaryToDec(number): 
    if number.find(" ") == -1:
        somma = 0
        number = number[::-1]
        for i in range(len(number)):
            somma += int(number[i]) * 2**i
            
        return somma
    else:
        return bitstring.BitArray(bin=number).float

#Normalize numbers with a range [0,range] in [0,1]
def normalize(number, range):
    return number/range

def inormalize(number, range):
    return int(number*255)

#Process the Image data for metrics
def ImageToFlattedArray(image):
    return np.ravel(np.asarray(image))

#Procedure to have two images of same size
def fixSizeImg(img, toFixImg, imgMode):
    mode = ("L" if imgMode == 1 else "1")
    width, heigth = im.imgSize(img)
    fWidth, fHeigth = im.imgSize(toFixImg)
    nImage = Image.new(mode, (width, heigth))
    for i in range(width):
        if i >= fWidth: break
        for j in range(heigth):
            if j >= fHeigth: break
            value = toFixImg.getpixel(xy=(i,j))
            nImage.putpixel(xy=(i,j),value=value)
    return nImage

    
