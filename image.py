from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np

#Return numpy array from a Image file
def loadImage(path):
    if path is "":
        print("Path must not be None!")
        return 

    img = Image.open(path)
    imgToArray = np.array(img)
    return imgToArray

#Show image from array or path
def showImage(img=None, path=None):
    if path is not None:
        Image.open(path).show()
    elif img is not None:
        Image.fromarray(img).show()
    else:
        print("showImage needs an array or a path!")

#Save image from array or Image file
def saveImage(img, path):
    if path is "":
        print("Path must not be None!")
        return 
    
    if type(img) is np.ndarray:
        Image.fromarray(img).save(path)
    else:
        img.save(path)

#Return image shape (width,heigth)
def getImageShape(img):
    return img.shape[1], img.shape[0]

#Arnold transform
def arnoldTransform(img):
    width, heigth = getImageShape(img)
    transformed = img.copy()

    for i in range(width):
        for j in range(heigth):
            newX = (i + j) % width
            newY = (i + 2*j) % heigth
            transformed[(newX,newY)] = img[(i,j)]

    return transformed

#Inverse Arnold transform
def iarnoldTransform(img):
    width, heigth = getImageShape(img)
    itransformed = img.copy()

    for i in range(width):
        for j in range(heigth):
            newX = (2*i - j) % width
            newY = (-i + j) % heigth
            itransformed[(newX,newY)] = img[(i,j)]

    return itransformed

#Esempio

img = loadImage("right.png")
t = arnoldTransform(img)
showImage(t)

it = iarnoldTransform(t)
showImage(it)