import numpy as np
import image_managing as im
import audio_managing as am
import watermark_embedding_extraction as watermark

#audio
AUDIO_PATH = "mono-piano.wav"
T_AUDIO_PATH = 0
T_SAMPLERATE = 1
LEN_FRAMES = 1024

#image
IMAGE_PATH = "right.png"

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
ALPHA = 0.001

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

def getStego(data, tupleAudio):
    nData = am.normalizeForWav(data)
    am.saveWavFile(tupleAudio[T_AUDIO_PATH], tupleAudio[T_SAMPLERATE], nData, "stego-magnitudo0001")     
    
def embedding(scramblingMode, imageMode, embeddingMode, frames = 0):
    #1 load audio file
    audioData, tupleAudio = getAudio(AUDIO_PATH)
    
    #2 run DWT on audio file
    DWTCoeffs = getDWT(audioData, WAVELET_TYPE, WAVELET_MODE)
    cA2, cD2, cD1 = DWTCoeffs
    
    #3 divide by frame
    if frames == 1:
        cA2 = am.audioToFrame(cA2, LEN_FRAMES)
        
    #4 run DCT on DWT coeffs   
    DCTCoeffs = am.DCT(cA2)
    
    #5 scrambling image watermark
    payload = getScrambling(IMAGE_PATH, SCRAMBLING_TECHNIQUES[scramblingMode], imageMode)
    
    #6 embedd watermark image
    if embeddingMode == "magnitudo":
        wCoeffs = watermark.magnitudoDCT(DCTCoeffs, payload, ALPHA)
    #print(wCoeffs)
    
    #7 run iDCT
    iWCoeffs = am.iDCT(wCoeffs)
    
    #8 join audio frames
    if frames == 1:
        iWCoeffs = am.frameToAudio(cA2)
    
    #9 run iDWT
    DWTCoeffs = iWCoeffs, cD2, cD1
    iWCoeffs = am.iDWT(DWTCoeffs, WAVELET_TYPE, WAVELET_MODE)
    
    #10 save new audio file
    getStego(iWCoeffs, tupleAudio)
        
if __name__ == "__main__":
    
    embedding(0, GRAYSCALE, "magnitudo")
    
    #print(type(dctCoeffs[0]))
    
    """
    img = im.loadImage("right.png")
    bimg = im.binarization(img)
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

