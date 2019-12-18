import numpy as np
import image_managing as im
import audio_managing as am
import watermark_embedding_extraction as watermark

if __name__ == "__main__":
    tupleAudio = am.readWavFile("mono-piano.wav")
    tupleAudio = am.audioData(tupleAudio)
    
    img = im.loadImage("right.png")
    bimg = im.binarization(img)
    gimg = im.grayscale(img)
    im.showImage(bimg)
    im.showImage(gimg)

    lsb = watermark.LSB(tupleAudio, bimg)
    ilsb = watermark.iLSB(lsb)
    im.showImage(ilsb)

    #Necessary to create mixed type array
    tupleAudio = np.asarray(tupleAudio, dtype=object)

    delta = watermark.deltaDCT(tupleAudio, gimg)
    idelta = watermark.ideltaDCT(tupleAudio, delta)
    im.showImage(idelta)
)

