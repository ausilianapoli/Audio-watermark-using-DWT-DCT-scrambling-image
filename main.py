import numpy as np
import image_managing as im
import audio_managing as am
import watermark_embedding_extraction as watermark

WAVELETS_LEVEL = 2

def getAudio(path):
    tupleAudio = am.readWavFile(path)
    audioData = am.audioData(tupleAudio)
    if am.isMono(audioData) == False:
        tupleAudio = am.joinAudioChannels(path)
        audioData = am.audioData(tupleAudio)
    return audioData

def getDWT(type, mode):
    waveletsFamilies = am.getWaveletsFamilies()
    DWTFamilies = am.filterWaveletsFamilies(waveletsFamilies)
    waveletsModes = am.getWaveletsModes()
    coeffs = am.DWT(audioData, DWTFamilies[DWTFamilies.index(type)], waveletsModes[waveletsModes.index(mode)], WAVELETS_LEVEL)
    cA2, cD2, cD1 = coeffs
    return cA2

if __name__ == "__main__":
    
    #1 load audio file
    audioData = getAudio("mono-piano.wav")
    
    #2 run DWT on audio file
    approxCoeffsDWT = getDWT("haar", "symmetric")
    
    #3 divide by frame
    
    #4 run DCT on DWT coeffs   
    DCTCoeffs = am.DCT(approxCoeffsDWT)
    
    #5 scrambling image watermark
    
    #6 embedd watermark image
    
    #7 run iDCT
    
    #8 join audio frames
    
    #9 run iDWT
    
    #10 save new audio file
    
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

