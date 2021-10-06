import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
from IPython.display import Audio
from scipy.io import wavfile
from scipy import linalg, hamming
from scipy.fft import fft, fftfreq


def plot_spectrograms(original, response, title, window='hanning', ylim=None):

    # Open both WAV files
    fs_o, y_o = wavfile.read(original)
    fs_r, y_r = wavfile.read(response)

    # Check both sample rates
    if fs_o != fs_r:
        raise Exception('Both sample rates must be equal!')

    fig, axs = plt.subplots(2, 1, figsize=(25,10), sharex=True)
    fig.suptitle(title, fontsize=20)

    # Original signal spectrogram plot
    f_o, t_o, sxx_o = sps.spectrogram(y_o, fs_o, window=window, nperseg=512, noverlap=1)
    axs[0].set_ylabel('Frequency [Hz]')
    axs[0].set_title('Original')

    # Response spectrogram plot
    f_r, t_r, sxx_r = sps.spectrogram(y_r, fs_r, window=window, nperseg=512, noverlap=1)
    axs[1].set_ylabel('Frequency [Hz]')
    axs[1].set_xlabel('Time [sec]')
    axs[1].set_title('Rebuilt')
    
    axs[0].specgram(y_o, Fs=fs_o)
    axs[1].specgram(y_r, Fs=fs_r)
    
    # y axis limit
    if (ylim != None):
            if (len(ylim) == 2):
                axs[1].set_ylim(ylim[0],ylim[1])
                axs[0].set_ylim(ylim[0],ylim[1])
                
def to_wav(track, filename, samplerate,  scale = 1):
    scaled_track = np.int16(track*32767*scale)
    wavfile.write(filename, samplerate, scaled_track)
    return scaled_track


def print_signal(x, y, title, samplerate=None, color='blue', xlim=None, window='hann', ylim=None):
    fig, axs = plt.subplots(2, 1, figsize=(25,10))
    fig.suptitle(title, fontsize=20)
    
    # Time plot
    axs[0].grid()
    axs[0].set_xlabel('Time [sec]')
    if (xlim != None):
        if (len(xlim) == 2):
            axs[0].set_xlim(xlim[0],xlim[1])
    axs[0].plot(x, y, color=color)
            
    # Spectrogram plot
    if samplerate != None:
        axs[1].set_ylabel('Frequency [Hz]')
        axs[1].set_xlabel('Time [sec]')
        if (ylim != None):
            if (len(ylim) == 2):
                axs[1].set_ylim(ylim[0],ylim[1])
        axs[1].specgram(y, Fs=samplerate)


