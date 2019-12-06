from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
from utils import imodule, coprime

#Return numpy array from a Image file
def loadImage(path=""):
    if path is "":
        print("LOAD IMAGE: Path must not be None!")
        return 

    img = Image.open(path)
    imgToArray = np.array(img)
    return imgToArray

#Show image from array or path
def showImage(img):
    if type(img) is np.ndarray:
        Image.fromarray(img).show()
    else:
        if img is "":
            print("SHOW IMAGE: Path must not be None!")
            return 
        Image.open(img).show()

#Save image from array or Image file
def saveImage(img, path):
    if path is "":
        print("SHOW IMAGE: Path must not be None!")
        return 
    
    if type(img) is np.ndarray:
        Image.fromarray(img).save(path)
    else:
        img.save(path)

#Return image shape (width,heigth)
def getImageShape(img):
    return img.shape[1], img.shape[0]

#Arnold transform
def arnoldTransform(img, iteration):
    width, heigth = getImageShape(img)
    if width != heigth:
        print("ARNOLD TRANSFORM: Image must be square!")
        return
    side = width
    toTransform = img.copy()
    transformed = img.copy()
    
    for iter in range(iteration):
        
        for i in range(side):
            for j in range(side):
                newX = (i + j) % side
                newY = (i + 2*j) % side
                transformed[(newY,newX)] = toTransform[(j,i)]
        toTransform = transformed.copy()

    return transformed


#Inverse Arnold transform
def iarnoldTransform(img, iteration):
    width, heigth = getImageShape(img)
    if width != heigth:
        print("IARNOLD TRANSFORM: Image must be square!")
        return
    side = width
    transformed = img.copy()
    toTransform = img.copy()
    
    for iter in range(iteration):
        
        for i in range(side):
            for j in range(side):
                newX = (2*i - j) % side
                newY = (-i + j) % side
                transformed[(newY,newX)] = toTransform[(j,i)]
        toTransform = transformed.copy()
    return transformed

#2D lower triangular mapping
def lowerTriangularMappingTransform(img, iteration, c, a=-1, d=-1):
    width, heigth = getImageShape(img)
    coprime_mode = "first"
    if a == -1 and d == -1:
        a = coprime(width, coprime_mode)
        d = coprime(heigth, coprime_mode)
    
    transformed = img.copy()
    toTransform = img.copy()
    
    for iter in range(iteration):
        
        for i in range(width):
            for j in range(heigth):
                newX = (a*i) % width
                newY = (c*i + d*j) % heigth
                transformed[(newY,newX)] = toTransform[(j,i)]

        toTransform = transformed.copy()
    
    return transformed
    
#2D inverse lower triangular mapping
def ilowerTriangularMappingTransform(img, iteration, c, a=-1, d=-1):
    width, heigth = getImageShape(img)
    coprime_mode = "first"
    if a == -1 and d == -1:
        a = coprime(width, coprime_mode)
        d = coprime(heigth, coprime_mode)
    
    transformed = img.copy()
    toTransform = img.copy()
    ia = imodule(a, width)
    id = imodule(d, heigth)
    for iter in range(iteration):
        
        for i in range(width):
            for j in range(heigth):
                newX = (ia*i) % width
                newY = (id*(j + (math.ceil(c*width/heigth)*heigth) - (c*newX))) % heigth
                transformed[(newY,newX)] = toTransform[(j,i)]

        toTransform = transformed.copy()
    return transformed

#2D upper triangular mapping
def upperTriangularMappingTransform(img, iteration, c, a=-1, d=-1):
    width, heigth = getImageShape(img)
    coprime_mode = "first"
    if a == -1 and d == -1:
        a = coprime(width, coprime_mode)
        d = coprime(heigth, coprime_mode)
    
    transformed = img.copy()
    toTransform = img.copy()
    
    for iter in range(iteration):
        
        for i in range(width):
            for j in range(heigth):
                newX = (a*i + c*j) % width
                newY = (d*j) % heigth
                transformed[(newY,newX)] = toTransform[(j,i)]

        toTransform = transformed.copy()
    
    return transformed
    
#2D inverse upper triangular mapping
def iupperTriangularMappingTransform(img, iteration, c, a=-1, d=-1):
    width, heigth = getImageShape(img)
    coprime_mode = "first"
    if a == -1 and d == -1:
        a = coprime(width, coprime_mode)
        d = coprime(heigth, coprime_mode)
    
    transformed = img.copy()
    toTransform = img.copy()
    ia = imodule(a, width)
    id = imodule(d, heigth)
    for iter in range(iteration):
        
        for i in range(width):
            for j in range(heigth):
                newY = (id*j) % heigth
                newX = (ia*(i + (math.ceil(c*heigth/width)*width) - (c*newY))) % width
                transformed[(newY,newX)] = toTransform[(j,i)]

        toTransform = transformed.copy()
    return transformed

def mappingTransform(mode, img, iteration, c, a=-1, d=-1):
    if mode is "lower":
        m = lowerTriangularMappingTransform(img,iteration,c,a,d)
        return m
    if mode is "upper":
        m = upperTriangularMappingTransform(img,iteration,c,a,d)
        return m
    else:
        print("MAPPING TRANSFORM: Mode must be lower or upper!")
        return

def imappingTransform(mode, img, iteration, c, a=-1, d=-1):
    if mode is "lower":
        m = ilowerTriangularMappingTransform(img,iteration,c,a,d)
        return m
    if mode is "upper":
        m = iupperTriangularMappingTransform(img,iteration,c,a,d)
        return m
    else:
        print("MAPPING TRANSFORM: Mode must be lower or upper!")
        return

'''
TESTING
'''

img = loadImage("right.png")
imgr = loadImage("07.jpg")

t = arnoldTransform(img,1)
showImage(t)

it = iarnoldTransform(t,1)
showImage(it)

m = mappingTransform(mode="lower",img=imgr,iteration=1,c=3,a=5)
showImage(m)
#saveImage(m, "triangular_2_iterations.png")

im = imappingTransform(mode="lower",img=m,iteration=1,c=3,a=5)
showImage(im)

m1 = mappingTransform(mode="upper",img=imgr,iteration=1,c=3)
showImage(m1)
#saveImage(m, "triangular_2_iterations.png")

im1 = imappingTransform(mode="upper",img=m1,iteration=1,c=3)
showImage(im1)
