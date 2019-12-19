import numpy as np
import image_managing as im
import audio_managing as am
import watermark_embedding_extraction as watermark

WAVELETS_LEVEL = 2

if __name__ == "__main__":
    tupleAudio = am.readWavFile("mono-piano.wav")
    audioData = am.audioData(tupleAudio)
    
    #Inglobare tutto questo in una singola funzione
    waveletsFamilies = am.getWaveletsFamilies()
    DWTFamilies = am.filterWaveletsFamilies(waveletsFamilies)
    waveletsModes = am.getWaveletsModes()
    coeffs = am.DWT(audioData, DWTFamilies[DWTFamilies.index("haar")], waveletsModes[waveletsModes.index("symmetric")], WAVELETS_LEVEL)
    cA2, cD2, cD1 = coeffs
    dctCoeff = am.DCT(cA2)
    print(type(dctCoeff[0]))
    
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

