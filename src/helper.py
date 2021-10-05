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
    axs[1].set_title('Response')
    
    axs[0].specgram(y_o, Fs=fs_o)
    axs[1].specgram(y_r, Fs=fs_r)
    
    # y axis limit
    if (ylim != None):
            if (len(ylim) == 2):
                axs[1].set_ylim(ylim[0],ylim[1])
                axs[0].set_ylim(ylim[0],ylim[1])
                
def to_wav(track, samplerate, filename, scale = 1):
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

        
# Assuming that response is delayed from the original. (Add exception!)
#
def sync_tracks(_original, response, scurity=500):
    if len(_original) > len(response):
        raise Exception('Response needs to be larger than original signal!')

    diff = len(response)-len(_original)
    # Original signal padding
    original = np.concatenate((_original, np.zeros(diff)))

    # Compute cross-correlation
    xcorr = sps.correlate(response / np.max(response), original / np.max(original), method='fft')

    # Calculate which lag to apply to sync signals
    sync_lag = np.abs((len(xcorr) // 2) - np.argmax(np.abs(xcorr))) - scurity
    return original, np.concatenate((response[sync_lag:], np.zeros(sync_lag)))

 
def get_r_p(original, recorded):
    N = len(original)
    
    # Calculate r (autocorrelation vector)
    r = sps.correlate(original, original, method='fft',mode='same')[N//2:] / N
    
    # Calculate p (cross-correlation vector) - Assuming both signals real and same size
    p = sps.correlate(recorded, original, method='fft',mode='same')[N//2:] / N
    #p = np.flip(p[:N//2])
    
    s2d = np.var(recorded)
    return r,p,s2d

def get_filter(r, p, M, sigma):
    r_ = r[:M]
    p_ = p[:M]
    wo = linalg.solve_toeplitz(r_, p_, check_finite=False)
    jo = sigma - p_.dot(wo)
    return wo, p_, jo, jo/sigma


def estimate_response(_original, _recorded, M, title):
    
    # Load files
    o_rate, original = wavfile.read(_original)
    r_rate, recorded = wavfile.read(_recorded) 
    
    # Normalize 
    original = original / 32767
    recorded = recorded / 32767

    # Substract mean
    original -= np.mean(original)
    recorded -= np.mean(recorded)
    
    # Aling both signals
    original_shifted, response_shifted = sync_tracks(original, recorded)
    
    # Estimate r, p and sigma^2
    r, p, sigma = get_r_p(original_shifted, response_shifted)
    
    # Estimate filter impulse response
    wo, p1, j, e  = get_filter(r, p, M, sigma)
    
    # Plot estimation
    fig, axs = plt.subplots(2, 1, figsize=(25,10))
    
    fig.suptitle(title, fontsize=25)
    
    # Print norm MSE
    print(f'epsilon = {e:.4f}')
    
    # Plot impulse response
    axs[0].grid(which='both', axis='both')
    axs[0].set_xlabel('Lags',fontsize=10)
    axs[0].set_title('Impulsive response estimation', fontsize=15, color='red')
    axs[0].scatter(np.arange(len(wo)), wo, color='red')

    # Compute and plot FFT
    freq = np.fft.fftfreq(len(wo), 1/r_rate)
    axs[1].set_title('Frequency response estimation', fontsize=15, color='blue')
    axs[1].set_xlim(10,24000)
    axs[1].grid(which='both', axis='both')
    axs[1].set_xlabel('Frequency [Hz]', fontsize=10)
    axs[1].set_ylabel('Magnitude [dB]', fontsize=10)
    axs[1].semilogx(freq[freq>0], 20*np.log10(np.abs(np.fft.fft(wo)))[freq>0], color='blue')


