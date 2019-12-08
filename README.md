# Audio-watermark-using-DWT-DCT-Arnold
Audio watermark using DWT, DCT and Arnold Transformation for embedding of binary image into audio signal.  
[Scientific Paper](https://ieeexplore.ieee.org/abstract/document/7150750)

## Tools
The follow tools are required:  
- scipy.io.wavfile;  
- subprocess;
- pywt;
- os;
- [ffmpeg-20191204](https://ffmpeg.zeranoe.com/builds/)

## Modules
Structure of the project:  
- `audio_managing.py` containing  
	- `readWavFile` to read file audio.wav from path;
	- `printMetadata` to print some information about audio (e.g. path, samplerate, #samples);
	- `isMono` to check the number of channels of input audio file;
	- `saveWavFile` to save processed file with audio wav format;
	- `joinAudioChannels` to join audio channels to only one;
	- `waveform` to plot the waveform of input audio file.
- `image_managing.py` containing  
- `watermark_embedding_extraction.py` containing
- `metrics.py` containing
- `attacks_on_watermark.py` containing
- `utils.py` containing
