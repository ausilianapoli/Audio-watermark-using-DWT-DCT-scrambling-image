from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

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

def imodule(a, m):
    a = a % m
    for x in range(m):
        if (a * x) % m == 1:
            return x   
    return 1

#Arnold transform
def arnoldTransform(img, iteration):
    side, _ = getImageShape(img)
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
    side, _ = getImageShape(img)
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
def triangularMappingTransform(img, iteration, a, c, d):
    width, heigth = getImageShape(img)
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
def itriangularMappingTransform(img, iteration, a, c, d):
    width, heigth = getImageShape(img)
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

#to write find numeri primi fra loro


'''
TESTING
'''
img = loadImage()
img = loadImage("right.png")
imgr = loadImage("07.jpg")

t = arnoldTransform(img,1)
showImage(t)

it = iarnoldTransform(t,1)
showImage(it)

m = triangularMappingTransform(imgr,2,1,10,1)
showImage(m)
saveImage(m, "triangular_2_iterations.png")

im = itriangularMappingTransform(m,2,1,10,1)
showImage(im)
