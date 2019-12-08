from utils import setLastBit, numberToBinary
from PIL import Image
from audio_managing import frameToAudio


def leastSignificantBit(audio, image):
    if image.mode is not "1":
        print("LEAST SIGNIFICANT BIT: Image must be binary!")
        return

    width, heigth = image.size
    audioLen = audio.shape[0]
    
    if width + heigth > audioLen:
        print("LEAST SIGNIFICANT BIT: Cover dimension is not sufficient for this payload size!")
        return

    if audio[0][0].shape[0] is not 1:
        joinAudio = frameToAudio(audio)

    bWidth = numberToBinary(width, 16)
    bHeigth = numberToBinary(heigth, 16)
    
    for i in range(width):
        for w in range(heigth):
            value = image.getpixel(xy=(i,j))
    """
    for frame in range(audioLen):
        frameLen = audio[i].shape[0]
        for i in range(frameLen):
            value = audio[frame][i]
            audio[frame][i] = setLastBit(value, )
    """