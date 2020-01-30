import os
import math
import numpy as np
import bitstring
import sys
from PIL import Image

#Return inverse module m of a
def imodule(a, m):
    if m == 0: return 0
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

#Routine to extract name of the file from its path without its extension
def withoutExtensionFile(pathname, sufix=""):
    pathname = os.path.basename(pathname).split(".")
    return pathname[0]+sufix

def splitFloat(number):
    if type(number) not in (float, np.float64):
        sys.exit("SPLIT FLOAT: number must be float!")
        return

    whole = int(number)
    dec = number - whole
    return whole, dec

def joinFloat(whole, dec):
    return whole + dec

def setLastBit(number, bit):
    if type(number) in (int, np.int16, np.int64):
        return ((number >> 1) << 1) | bit
    elif type(number) in (float, np.float64):
        whole, dec = splitFloat(number)
        number = setLastBit(whole, bit)
        return joinFloat(number, dec)
        """
        binary = decToBinary(number)
        binary = binary[:23] + str(bit) + binary[24:]
        return binaryToDec(binary)
        """

def getLastBit(number):
    if type(number) in (int, np.int16, np.int64):
        return int(number % 2)
    elif type(number) in (float, np.float64):
        whole, dec = splitFloat(number)
        return getLastBit(whole)

        """
        binary = decToBinary(number)
        return int(binary[23])
        """

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
    width, heigth = img.size
    fWidth, fHeigth = toFixImg.size
    nImage = Image.new(mode, (width, heigth))
    for i in range(width):
        if i >= fWidth: break
        for j in range(heigth):
            if j >= fHeigth: break
            value = toFixImg.getpixel(xy=(i,j))
            nImage.putpixel(xy=(i,j),value=value)
    return nImage

def setBinary(coeff, bit):
    if bit == 255:
        coeff[-1] = 10
    else:
        coeff[-1] = -10

    return coeff
    
def getBinary(coeff):
    if coeff[-1] > 0:
        return 255
    else:
        return 0 

def setGray(coeff, bit):
    coeff[-1] = bit - 127

    return coeff
    
def getGray(coeff):
    return int(coeff[-1] + 127)

def setDelta(norm, delta, value):
    if value == 255:
        return (norm + delta), (norm - delta)
    else:
        return (norm - delta), (norm + delta)

def getDelta(norm1 , norm2):
    if norm1 > norm2:
        return 255
    else:
        return 0

def subVectors(coeff):
    coeffsLen = len(coeff) // 2
    v1 = np.zeros(coeffsLen)
    v2 = np.zeros(coeffsLen)

    for i in range(coeffsLen):
        v1[i] = coeff[2*(i+1) - 1]
        v2[i] = coeff[2*(i+1) - 2]

    return v1, v2

def isubVectors(v1, v2):
    coeffsLen = len(v1)
    v = np.zeros(coeffsLen * 2)

    for i in range(coeffsLen):
        v[2*(i+1) - 1] = v1[i]
        v[2*(i+1) - 2] = v2[i]
    
    return v

def inormCalc(norm1, u1):
    vLen = len(u1)
    v1 = np.zeros(vLen)

    for i in range(vLen):
        v1[i] = norm1 * u1[i]

    return v1

def normCalc(v):
    norm = 0
    for coeff in v:
        norm += coeff**2
    norm = math.sqrt(norm)
    u = v/norm
    return norm, u
