from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np

#Return numpy array from a Image file
def loadImage(path=""):
    if path is "":
        print("LOAD IMAGE: Path must not be None!")
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
        print("SHOW IMAGE: showImage needs an array or a path!")

#Save image from array or Image file
def saveImage(img, path):
    if path is "":
        print("SAVE IMAGE: Path must not be None!")
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
    toTransform = img.copy()
    transformed = img.copy()
    
    for iter in range(iteration):
        
        for i in range(width):
            for j in range(heigth):
                newX = (i + j) % width
                newY = (i + 2*j) % heigth
                transformed[(newX,newY)] = toTransform[(i,j)]
        toTransform = transformed.copy()

    return transformed


#Inverse Arnold transform
def iarnoldTransform(img, iteration):
    width, heigth = getImageShape(img)
    itransformed = img.copy()
    itoTransform = img.copy()
    
    for iter in range(iteration):
        
        for i in range(width):
            for j in range(heigth):
                newX = (2*i - j) % width
                newY = (-i + j) % heigth
                itransformed[(newX,newY)] = itoTransform[(i,j)]
        itoTransform = itransformed.copy()
    return itransformed


'''
TESTING
'''

img = loadImage()
img = loadImage("right.png")
t = arnoldTransform(img,5)
saveImage(t, "here.png")
showImage(t)
it = iarnoldTransform(t,5)
showImage(it)
