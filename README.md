# Audio Watermark using DWT-DCT approach and scrambling image
Audio watermark using DWT-DCT approach and scrambling methods for embedding of binary or grayscale image into audio signal.  
[Scientific Paper](https://ieeexplore.ieee.org/abstract/document/7150750)

## Tools
The follow specific tools are required:  
- scipy.io.wavfile;  
- scipy.fftpack
- pywt;
- PIL;
- [ffmpeg-20191204](https://ffmpeg.zeranoe.com/builds/)

## Modules
Structure of the project:  
- `audio_managing.py` containing functions to read, write wave files and to apply DWT, DCT and FFT;
- `image_managing.py` containing functions to read, write images and to apply scrambling procedures;
- `watermark_embedding_extraction.py` containing various techniques for binary and grayscale images;
- `metrics.py` containing PSNR, SNR and Pearson's Index to evaluate techniques' performances;
- `attacks_on_watermark.py` containing some attacks to audio signal;
- `utils.py` containing various functions of general purpose;
- `main.py` containing the pipeline of the whole project.

 
