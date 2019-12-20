import scipy.stats as stats
import numpy as np
import math

#Pearson index to "view" the correlation between two watermarks
def correlationIndex(wOriginal, wExtracted):
    p = stats.pearsonr(wOriginal, wExtracted)
    return p

#Binary Detection of corretc watermark based on threshold
def binaryDetection(index, threshold):
    return (True if abs(index[0]) > threshold else False)

#PSNR
def PSNR(wOriginal, wExtracted):
    mse = np.mean((wOriginal - wExtracted)**2)
    psnr = 10 * math.log10((255.0**2)/mse)
    return psnr

'''
TESTING
'''

'''
image1 = Image.new("L",(3,4))
image1.putpixel(xy=(1,2),value=255)
image1.putpixel(xy=(2,2),value=100)
image1.putpixel(xy=(1,0),value=150)
image1.putpixel(xy=(2,3),value=55)
image1.putpixel(xy=(0,0),value=70)
array1 = np.ravel(np.asarray(image1))


image2 = Image.new("L",(3,4))
image2.putpixel(xy=(1,2),value=25)
image2.putpixel(xy=(2,2),value=10)
image2.putpixel(xy=(1,0),value=149)
image2.putpixel(xy=(2,3),value=50)
image2.putpixel(xy=(0,0),value=40)
array2 = np.ravel(np.asarray(image2))

p = correlationIndex(array1, array2)
print(p)
print(binaryDetection(p, 0.7))
'''