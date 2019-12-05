import os

def makeFileName(prefix, path):
    fileName = os.path.basename(path)
    dirName = os.path.dirname(path)
    fileName = prefix + "-" + fileName
    nPath = os.path.join(dirName, fileName)
    return nPath
