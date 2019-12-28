import numpy as np
import image_managing as im
import audio_managing as am
import watermark_embedding_extraction as watermark
from utils import makeFileName, ImageToFlattedArray, fixSizeImg
import metrics as m
import attacks as a

#audio
T_AUDIO_PATH = 0
T_SAMPLERATE = 1
LEN_FRAMES = 16

#DWT
WAVELETS_LEVEL = 1
WAVELET_TYPE = "db1"
WAVELET_MODE = "symmetric"

#scrambling
SCRAMBLING_TECHNIQUES = ["arnold", "lower", "upper"]
BINARY = 0
GRAYSCALE = 1
NO_ITERATIONS = 1
TRIANGULAR_PARAMETERS = [5, 3, 1] #c,a,d

#embedding
ALPHA = 0.1

#attacks
CUTOFF_FREQUENCY = 22050

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
    #cA, cD2, cD1 = DWTCoeffs #level 2
    cA, cD1 = DWTCoeffs #level 1

    #3 divide by frame & #4 run DCT on DWT coeffs
    if frames == 1:
        cA = am.audioToFrame(cA, LEN_FRAMES)
        DCTCoeffs = np.copy(cA)

        for i in range(cA.shape[0]):
            DCTCoeffs[i] = am.DCT(cA[i])
    #4 run DCT on DWT coeffs   
    else:
        DCTCoeffs = am.DCT(cA)

    #5 scrambling image watermark
    payload = getScrambling(imagePath, SCRAMBLING_TECHNIQUES[scramblingMode], imageMode)
 
    #6 embedd watermark image
    if embeddingMode == "magnitudo":
        wCoeffs = watermark.magnitudoDCT(DCTCoeffs, payload, ALPHA)
    elif embeddingMode == "lsb":
        wCoeffs = watermark.LSB(DCTCoeffs, payload)
    elif embeddingMode == "delta":
        wCoeffs = watermark.deltaDCT(DCTCoeffs, payload)
    elif embeddingMode == "bruteBinary":
        wCoeffs = watermark.bruteBinary(DCTCoeffs, payload)
    elif embeddingMode == "bruteGray":
        wCoeffs = watermark.bruteGray(DCTCoeffs, payload)

    #7 run iDCT and #8 join audio frames
    if frames == 1:
        iWCoeffs = np.copy(wCoeffs)
        for i in range(wCoeffs.shape[0]):
            iWCoeffs[i] = am.iDCT(wCoeffs[i])

        iWCoeffs = am.frameToAudio(iWCoeffs)

    #7 run iDCT
    else:
        iWCoeffs = am.iDCT(wCoeffs)

    
    #9 run iDWT
    #DWTCoeffs = iWCoeffs, cD2, cD1 #level 2
    DWTCoeffs = iWCoeffs, cD1 #level 1
    iWCoeffs = am.iDWT(DWTCoeffs, WAVELET_TYPE, WAVELET_MODE)

    #10 save new audio file
    getStego(iWCoeffs, tupleAudio, outputAudioPath)

    return wCoeffs #return information for extraction
    
def extraction(stegoAudio, audio, outputImagePath, scramblingMode, embeddingMode, frames = 0):
    #1 load audio file
    audioData, tupleAudio = getAudio(audio)
    stegoAudioData, stegoTupleAudio = getAudio(stegoAudio)

    #2 run DWT on audio file
    DWTCoeffs = getDWT(audioData, WAVELET_TYPE, WAVELET_MODE)
    #cA, cD2, cD1 = DWTCoeffs #level 2
    cA, cD1 = DWTCoeffs #level 1

    stegoDWTCoeffs = getDWT(stegoAudioData, WAVELET_TYPE, WAVELET_MODE)
    #stegocA2, stegocD2, stegocD1 = stegoDWTCoeffs #level 2
    stegocA, stegocD1 = stegoDWTCoeffs #level 1
    
    #3 divide by frame & #4 run DCT on DWT coeffs
    if frames == 1:
        cA = am.audioToFrame(cA, LEN_FRAMES)
        DCTCoeffs = np.copy(cA)
        for i in range(cA.shape[0]):
            DCTCoeffs[i] = am.DCT(cA[i])
        
        stegocA = am.audioToFrame(stegocA, LEN_FRAMES)
        stegoDCTCoeffs = np.copy(stegocA)
        for i in range(stegocA.shape[0]):
            stegoDCTCoeffs[i] = am.DCT(stegocA[i])

    #4 run DCT on DWT coeffs   
    else:
        DCTCoeffs = am.DCT(cA)
        stegoDCTCoeffs = am.DCT(stegocA)
        
    #print("DCTCoeffs: ", DCTCoeffs)
    #print("StegoDCTCoeffs: ", stegoDCTCoeffs)

    #5 extract image watermark
    if embeddingMode == "magnitudo":
        payload = watermark.imagnitudoDCT(DCTCoeffs, stegoDCTCoeffs, ALPHA)
    elif embeddingMode == "lsb":
        payload = watermark.iLSB(stegoDCTCoeffs)
    elif embeddingMode == "delta":
        payload = watermark.ideltaDCT(stegoDCTCoeffs)
    elif embeddingMode == "bruteBinary":
        payload = watermark.ibruteBinary(stegoDCTCoeffs)
    elif embeddingMode == "bruteGray":
        payload = watermark.ibruteGray(stegoDCTCoeffs)
    
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

def compareAudio(audio, stegoAudio):
    audio = am.audioData(am.readWavFile(audio))
    stegoAudio = am.audioData(am.readWavFile(stegoAudio))
    snr = m.SNR(audio)
    snrStego = m.SNR(stegoAudio)
    return snr, snrStego

def attackStego(stegoAudio):
    stegoAudio = am.readWavFile(stegoAudio)
    tAmplitude = [0.5, 2]
    for i in range(len(tAmplitude)):
        getStego(a.amplitudeScaling(stegoAudio[2], tAmplitude[i]), stegoAudio, "amplitude{}".format(tAmplitude[i]))
    sampleRates = [int(stegoAudio[T_SAMPLERATE]*0.75), int(stegoAudio[T_SAMPLERATE]*0.5), int(stegoAudio[T_SAMPLERATE]*0.25)]
    for i in range(len(sampleRates)):
        a.resampling(stegoAudio[T_AUDIO_PATH], sampleRates[i])
    nLPFilter = [2, 4, 6]
    tupleFFT = am.FFT(stegoAudio)
    indexCutoff = am.indexFrequency(tupleFFT[1], stegoAudio[T_SAMPLERATE], CUTOFF_FREQUENCY)
    for i in range(len(nLPFilter)):
        getStego(am.iFFT(a.butterLPFilter(tupleFFT[0], indexCutoff, nLPFilter[i])), stegoAudio, "butter{}".format(nLPFilter[i]))
    sigmaGauss = [0.00005, 0.0001, 0.00015, 0.0002]
    for i in range(len(sigmaGauss)):
        getStego(a.gaussianNoise(am.audioData(stegoAudio), sigmaGauss[i]), stegoAudio, "gauss{}".format(sigmaGauss[i]))

if __name__ == "__main__":
    
    #wCoeffs = embedding("mono-piano.wav", "right.png", "stego-magnitudo01", 2, GRAYSCALE, "magnitudo", 1)
    #wCoeffs = embedding("mono-piano.wav", "right.png", "stego-lsb", 0, BINARY, "lsb")
    #wCoeffs = embedding("buddy.wav", "right.png", "stego-binary-brute", 1, BINARY, "bruteBinary",1)
    #wCoeffs = embedding("buddy.wav", "right.png", "stego-gray-brute", 1, GRAYSCALE, "bruteGray",1)
    wCoeffs = embedding("buddy.wav", "right.png", "stego-binary-delta", 0, BINARY, "delta",1)
    #print(wCoeffs)
    
    #extraction("stego-magnitudo001-mono-piano.wav", "mono-piano.wav", "magnitudo001-right.png", 2, "magnitudo", 1)
    #extraction("stego-lsb-mono-piano.wav", "mono-piano.wav", "lsb-right.png", 0, "lsb")
    #extraction("stego-binary-brute-buddy.wav", "buddy.wav", "brute-binary-right.png", 1, "bruteBinary",1)
    #extraction("stego-gray-brute-buddy.wav", "buddy.wav", "brute-gray-right.png", 1, "bruteGray",1)
    extraction("stego-binary-delta-buddy.wav", "buddy.wav", "delta-binary-right.png", 0, "delta",1)
    """
    result = compareWatermark("right.png", "delta-grayscale-right.png", GRAYSCALE)
    print("The extracted watermark is correlated to that original? ", result[0])
    print("The PSNR between the two watermarks is: ", result[1])
    snr = compareAudio("mono-piano.wav", "stego-grayscale-delta-mono-piano.wav")
    print("SNR of {} is: {}\nSNR of {} is: {}".format("mono-piano.wav", snr[0], "stego-grayscale-delta-mono-piano.wav", snr[1]))
    
    attackStego("stego-grayscale-delta-mono-piano.wav")
    extraction("12000-stego-grayscale-delta-mono-piano.wav", "stego-grayscale-delta-mono-piano.wav", "12000-delta-grayscale-right.png", 0, "delta",1)
    extraction("24000-stego-grayscale-delta-mono-piano.wav", "stego-grayscale-delta-mono-piano.wav", "24000-delta-grayscale-right.png", 0, "delta",1)
    extraction("36000-stego-grayscale-delta-mono-piano.wav", "stego-grayscale-delta-mono-piano.wav", "36000-delta-grayscale-right.png", 0, "delta",1)
    extraction("amplitude0.5-stego-grayscale-delta-mono-piano.wav", "stego-grayscale-delta-mono-piano.wav", "amplitude0.5-delta-grayscale-right.png", 0, "delta",1)
    extraction("amplitude2-stego-grayscale-delta-mono-piano.wav", "stego-grayscale-delta-mono-piano.wav", "amplitude2-delta-grayscale-right.png", 0, "delta",1)
    extraction("butter2-stego-grayscale-delta-mono-piano.wav", "stego-grayscale-delta-mono-piano.wav", "butter2-delta-grayscale-right.png", 0, "delta",1)
    extraction("butter4-stego-grayscale-delta-mono-piano.wav", "stego-grayscale-delta-mono-piano.wav", "butter4-delta-grayscale-right.png", 0, "delta",1)
    extraction("butter6-stego-grayscale-delta-mono-piano.wav", "stego-grayscale-delta-mono-piano.wav", "butter6-delta-grayscale-right.png", 0, "delta",1)
    extraction("gauss0.0001-stego-grayscale-delta-mono-piano.wav", "stego-grayscale-delta-mono-piano.wav", "gauss0.0001-delta-grayscale-right.png", 0, "delta",1)
    extraction("gauss0.0002-stego-grayscale-delta-mono-piano.wav", "stego-grayscale-delta-mono-piano.wav", "gauss0.0002-delta-grayscale-right.png", 0, "delta",1)
    extraction("gauss0.00015-stego-grayscale-delta-mono-piano.wav", "stego-grayscale-delta-mono-piano.wav", "gauss0.00015-delta-grayscale-right.png", 0, "delta",1)
    extraction("gauss5e-05-stego-grayscale-delta-mono-piano.wav", "stego-grayscale-delta-mono-piano.wav", "gauss5e-05-delta-grayscale-right.png", 0, "delta",1)
    result = compareWatermark("delta-grayscale-right.png", "12000-delta-grayscale-right.png", GRAYSCALE)
    print("The extracted watermark is correlated to that original? ", result[0])
    print("The PSNR between the two watermarks is: ", result[1])
    result = compareWatermark("delta-grayscale-right.png", "24000-delta-grayscale-right.png", GRAYSCALE)
    print("The extracted watermark is correlated to that original? ", result[0])
    print("The PSNR between the two watermarks is: ", result[1])
    result = compareWatermark("delta-grayscale-right.png", "36000-delta-grayscale-right.png", GRAYSCALE)
    print("The extracted watermark is correlated to that original? ", result[0])
    print("The PSNR between the two watermarks is: ", result[1])
    result = compareWatermark("delta-grayscale-right.png", "amplitude0.5-delta-grayscale-right.png", GRAYSCALE)
    print("The extracted watermark is correlated to that original? ", result[0])
    print("The PSNR between the two watermarks is: ", result[1])
    result = compareWatermark("delta-grayscale-right.png", "amplitude2-delta-grayscale-right.png", GRAYSCALE)
    print("The extracted watermark is correlated to that original? ", result[0])
    print("The PSNR between the two watermarks is: ", result[1])
    result = compareWatermark("delta-grayscale-right.png", "butter2-delta-grayscale-right.png", GRAYSCALE)
    print("The extracted watermark is correlated to that original? ", result[0])
    print("The PSNR between the two watermarks is: ", result[1])
    result = compareWatermark("delta-grayscale-right.png", "butter4-delta-grayscale-right.png", GRAYSCALE)
    print("The extracted watermark is correlated to that original? ", result[0])
    print("The PSNR between the two watermarks is: ", result[1])
    result = compareWatermark("delta-grayscale-right.png", "butter6-delta-grayscale-right.png", GRAYSCALE)
    print("The extracted watermark is correlated to that original? ", result[0])
    print("The PSNR between the two watermarks is: ", result[1])
    result = compareWatermark("delta-grayscale-right.png", "gauss0.0001-delta-grayscale-right.png", GRAYSCALE)
    print("The extracted watermark is correlated to that original? ", result[0])
    print("The PSNR between the two watermarks is: ", result[1])
    result = compareWatermark("delta-grayscale-right.png", "gauss0.0002-delta-grayscale-right.png", GRAYSCALE)
    print("The extracted watermark is correlated to that original? ", result[0])
    print("The PSNR between the two watermarks is: ", result[1])
    result = compareWatermark("delta-grayscale-right.png", "gauss0.00015-delta-grayscale-right.png", GRAYSCALE)
    print("The extracted watermark is correlated to that original? ", result[0])
    print("The PSNR between the two watermarks is: ", result[1])
    result = compareWatermark("delta-grayscale-right.png", "gauss5e-05-delta-grayscale-right.png", GRAYSCALE)
    print("The extracted watermark is correlated to that original? ", result[0])
    print("The PSNR between the two watermarks is: ", result[1])
    """
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

