import os
import math

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
        print("COPRIME: Mode must be first or second!")
        return

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

#Routine to create the pathname for file before saving
def makeFileName(prefix, path):
    fileName = os.path.basename(path)
    dirName = os.path.dirname(path)
    fileName = prefix + "-" + fileName
    nPath = os.path.join(dirName, fileName)
    return nPath

def setLastBit(number, bit):
    return ((number >> 1) << 1) | bit

def getLastBit(number):
    return int(number % 2)

#Return a string containing the number written in binary notation with bits bit
def decToBinary(number, bits=16):
    if number > 2**bits:
        print("DEC TO BINARY: Insufficient number of bit!")
        return

    binary = ""
    for i in range(bits):
        if number != 0:
            bit = number % 2
            number = number >> 1
            binary += str(bit)
        else:
            binary += str(0)
    
    return binary[::-1]

#Return a int from a string containing the number written in binary notation
def binaryToDec(number):
    somma = 0
    number = number[::-1]
    for i in range(len(number)):
        somma += int(number[i]) * 2**i
        
    return somma

#Normalize numbers with a range [0,range] in [0,1]
def normalize(number, range):
    return number/range

def inormalize(number, range):
    return int(number*255)
