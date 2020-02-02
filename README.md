# Audio Watermark using DWT-DCT approach and scrambling image
Audio watermark using DWT-DCT approach and scrambling methods for embedding of binary or grayscale image into audio signal. Its goal is to guarantee authenticity of the audio signal.  
This is our project for the academic course of Multimedia at University of Catania for the Master Degree in Computer Science.  
The scientific papers to which we referred are:
- [Blind Audio Watermarking Based On Discrete Wavelet and Cosine Transform](https://ieeexplore.ieee.org/abstract/document/7150750);
- [Novel secured scheme for blind audio/speech norm-space watermarking by Arnold algorithm](https://www.sciencedirect.com/science/article/pii/S016516841830272X);
- [2D Triangular Mappings and Their Applications in Scrambling Rectangle Image](https://www.researchgate.net/publication/26557013_2D_Triangular_Mappings_and_Their_Applications_in_Scrambling_Rectangle_Image).

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
- `main.py` containing the pipeline of the whole project and in the txt file `commands_to_run` there are the commands for bash.

## Authors
- [Maria Ausilia Napoli Spatafora](https://github.com/ausilianapoli)
- [Mattia Litrico](https://github.com/mattia1997)
 
