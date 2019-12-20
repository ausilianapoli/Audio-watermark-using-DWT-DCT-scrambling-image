import numpy as np
import image_managing as im
import audio_managing as am
import watermark_embedding_extraction as watermark
from utils import makeFileName, ImageToFlattedArray, fixSizeImg
import metrics as m

#audio
T_AUDIO_PATH = 0
T_SAMPLERATE = 1
LEN_FRAMES = 1024

#DWT
WAVELETS_LEVEL = 2
WAVELET_TYPE = "haar"
WAVELET_MODE = "symmetric"

#scrambling
SCRAMBLING_TECHNIQUES = ["arnold", "lower", "upper"]
BINARY = 0
GRAYSCALE = 1
NO_ITERATIONS = 1
TRIANGULAR_PARAMETERS = [5, 3, 1] #c,a,d

#embedding
ALPHA = 0.1

def getAudio(path):
    tupleAudio = am.readWavFile(path)
    audioData = am.audioData(tupleAudio)
    if am.isMono(audioData) == False:
        tupleAudio = am.joinAudioChannels(path)
        audioData = am.audioData(tupleAudio)
    return audioData, tupleAudio

def getDWT(audioData, type, mode):
    waveletsFamilies = am.getWaveletsFamilies()
    DWTFamilies = am.filterWaveletsFamilies(waveletsFamilies)
    waveletsModes = am.getWaveletsModes()
    coeffs = am.DWT(audioData, DWTFamilies[DWTFamilies.index(type)], waveletsModes[waveletsModes.index(mode)], WAVELETS_LEVEL)
    return coeffs

def getScrambling(path, type, mode = BINARY):
    image = im.loadImage(path)
    if mode == BINARY:
        image = im.binarization(image)
    else:
        image = im.grayscale(image)
    if type == "arnold":
        image = im.arnoldTransform(image, NO_ITERATIONS)
    elif type == "lower" or type == "upper":
        image = im.mappingTransform(type, image, NO_ITERATIONS, TRIANGULAR_PARAMETERS[0], TRIANGULAR_PARAMETERS[1], TRIANGULAR_PARAMETERS[2])
    return image

def getiScrambling(payload, type):
    if type == "arnold":
        image = im.iarnoldTransform(payload, NO_ITERATIONS)
    elif type == "lower" or type == "upper":
        image = im.imappingTransform(type, payload, NO_ITERATIONS, TRIANGULAR_PARAMETERS[0], TRIANGULAR_PARAMETERS[1], TRIANGULAR_PARAMETERS[2])
    return image

def getStego(data, tupleAudio, outputAudioPath):
    nData = am.normalizeForWav(data)
    am.saveWavFile(tupleAudio[T_AUDIO_PATH], tupleAudio[T_SAMPLERATE], nData, outputAudioPath)

def getPayload(image, outputImagePath):
    #fileName = makeFileName(outputImageName, outputImagePath)
    im.saveImage(image, outputImagePath)
    
def embedding(audioPath, imagePath, outputAudioPath, scramblingMode, imageMode, embeddingMode, frames = 0):
    #1 load audio file
    audioData, tupleAudio = getAudio(audioPath)
    
    #2 run DWT on audio file
    DWTCoeffs = getDWT(audioData, WAVELET_TYPE, WAVELET_MODE)
    cA2, cD2, cD1 = DWTCoeffs
    
    #3 divide by frame & #4 run DCT on DWT coeffs
    if frames == 1:
        cA2 = am.audioToFrame(cA2, LEN_FRAMES)
        DCTCoeffs = np.copy(cA2)
        #print(cA2[0][0])
        for i in range(cA2.shape[0]):
            DCTCoeffs[i] = am.DCT(cA2[i])
    #4 run DCT on DWT coeffs   
    else:
        DCTCoeffs = am.DCT(cA2)
    
    #print(DCTCoeffs)
    
    #5 scrambling image watermark
    payload = getScrambling(imagePath, SCRAMBLING_TECHNIQUES[scramblingMode], imageMode)
    
    #6 embedd watermark image
    if embeddingMode == "magnitudo":
        wCoeffs = watermark.magnitudoDCT(DCTCoeffs, payload, ALPHA)
    elif embeddingMode == "lsb":
        wCoeffs = watermark.LSB(DCTCoeffs, payload)
    elif embeddingMode == "delta":
        wCoeffs = watermark.deltaDCT(DCTCoeffs, payload)
        print(wCoeffs)
    
    #7 run iDCT and #8 join audio frames
    if frames == 1:
        iWCoeffs = np.copy(wCoeffs)
        for i in range(wCoeffs.shape[0]):
            iWCoeffs[i] = am.iDCT(wCoeffs[i])
        iWCoeffs = am.frameToAudio(cA2)
    #7 run iDCT
    else:
        iWCoeffs = am.iDCT(wCoeffs)
        
    #9 run iDWT
    DWTCoeffs = iWCoeffs, cD2, cD1
    iWCoeffs = am.iDWT(DWTCoeffs, WAVELET_TYPE, WAVELET_MODE)
    
    #10 save new audio file
    getStego(iWCoeffs, tupleAudio, outputAudioPath)
    
    return wCoeffs #return information for extraction
    
def extraction(stegoAudio, audio, outputImagePath, scramblingMode, embeddingMode, frames = 0):
    #1 load audio file
    audioData, tupleAudio = getAudio(audio)
    #stegoAudioData, stegoTupleAudio = getAudio(stegoAudio)
    
    #2 run DWT on audio file
    DWTCoeffs = getDWT(audioData, WAVELET_TYPE, WAVELET_MODE)
    cA2, cD2, cD1 = DWTCoeffs

    #stegoDWTCoeffs = getDWT(stegoAudioData, WAVELET_TYPE, WAVELET_MODE)
    #stegocA2, stegocD2, stegocD1 = stegoDWTCoeffs
    
    #3 divide by frame & #4 run DCT on DWT coeffs
    if frames == 1:
        cA2 = am.audioToFrame(cA2, LEN_FRAMES)
        DCTCoeffs = np.copy(cA2)
        for i in range(cA2.shape[0]):
            DCTCoeffs[i] = am.DCT(cA2[i])
        #stegocA2 = am.audioToFrame(stegocA2, LEN_FRAMES)
        #stegoDCTCoeffs = np.copy(stegocA2)
        #for i in range(stegocA2.shape[0]):
            #stegoDCTCoeffs[i] = am.DCT(stegocA2)
    #4 run DCT on DWT coeffs   
    else:
        DCTCoeffs = am.DCT(cA2)
        #stegoDCTCoeffs = am.DCT(stegocA2)
    
    #5 extract image watermark
    if embeddingMode == "magnitudo":
        payload = watermark.imagnitudoDCT(DCTCoeffs, stegoAudio, ALPHA)
    elif embeddingMode == "lsb":
        payload = watermark.iLSB(stegoAudio)
    elif embeddingMode == "delta":
        payload = watermark.ideltaDCT(DCTCoeffs, stegoAudio)
    
    #6 inverse scrambling of payload
    payload = getiScrambling(payload, SCRAMBLING_TECHNIQUES[scramblingMode])
    
    #7 save image
    getPayload(payload, outputImagePath)
    
def compareWatermark(wOriginal, wExtracted, imgMode):
    wOriginal = im.loadImage(wOriginal)
    if imgMode == "GRAYSCALE":
        wOriginal = im.grayscale(wOriginal)
    else:
        wOriginal = im.binarization(wOriginal)
    wExtracted = im.loadImage(wExtracted)
    #print(im.imgSize(wOriginal), im.imgSize(wExtracted))
    if(im.imgSize(wOriginal) != im.imgSize(wExtracted)):
        wExtracted = fixSizeImg(wOriginal, wExtracted, imgMode)
    wOriginal = ImageToFlattedArray(wOriginal)
    wExtracted = ImageToFlattedArray(wExtracted)
    #print(len(wOriginal), len(wExtracted))
    p = m.correlationIndex(wOriginal, wExtracted)
    psnr = m.PSNR(wOriginal, wExtracted)
    return m.binaryDetection(p, 0.7), psnr
        
if __name__ == "__main__":
    
    wCoeffs = embedding("mono-piano.wav", "right.png", "stego-magnitudo01", 2, GRAYSCALE, "magnitudo", 1)
    #wCoeffs = embedding("mono-piano.wav", "right.png", "stego-lsb", 0, BINARY, "lsb")
    #wCoeffs = embedding("mono-piano.wav", "right.png", "stego-delta", 0, GRAYSCALE, "delta")

    #print(wCoeffs)
    
    extraction(wCoeffs, "mono-piano.wav", "magnitudo01-right.png", 2, "magnitudo", 1)
    #extraction("stego-lsb-mono-piano.wav", "mono-piano.wav", "lsb-right.png", 0, "lsb")
    #extraction("stego-delta-mono-piano.wav", "mono-piano.wav", "delta-right.png", 0, "delta")
    
    result = compareWatermark("right.png", "magnitudo01-right.png", GRAYSCALE)
    print("The extracted watermark is correlated to that original? ", result[0])
    print("The PSNR between the two watermarks is: ", result[1])
    
    """
    img = im.loadImage("right.png")
    bimg = im.binarization(img)
    bimg.show()
    
    abimg = im.arnoldTransform(bimg, 1)
    gimg = im.grayscale(img)
    agimg = im.arnoldTransform(gimg, 1)
    im.showImage(bimg)
    im.showImage(gimg)

    lsb = watermark.LSB(audioData, abimg)
    ilsb = watermark.iLSB(lsb)
    ilsb = im.iarnoldTransform(ilsb, 1)
    im.showImage(ilsb)

    #Necessary to create mixed type array
    audioData = np.asarray(audioData, dtype=object)

    delta = watermark.deltaDCT(audioData, agimg)
    idelta = watermark.ideltaDCT(audioData, delta)
    idelta = im.iarnoldTransform(idelta, 1)
    im.showImage(idelta)
    """

