import numpy
import scipy.io.wavfile
from scipy.fftpack import dct


import torch
from torch import nn, optim
import torch.nn.functional as F

from matplotlib.cbook import (
    MatplotlibDeprecationWarning, sanitize_sequence)
from matplotlib.cbook import mplDeprecation  # deprecated
from matplotlib.rcsetup import defaultParams, validate_backend, cycler
import matplotlib.pyplot as plt
import librosa
import librosa.display


def create_chroma_filters(sample_rate=16000, n_fft=512, n_chroma=12):
    # -------------------------------
    # Chroma Filter Bank from Librosa
    # -------------------------------
    #fbank = numpy.zeros((nfilt, int(numpy.floor(n_fft / 2 + 1))))
    fbank = librosa.filters.chroma(sr=sample_rate, n_fft=n_fft, n_chroma=n_chroma)
    
    #print(f'shape of fbank = {fbank.shape}.   .T = {fbank.T.shape}')

    #plt.figure(figsize=(10, 4));
    
    #idxs_to_plot = [0, 3, 6, n_chroma-4]
    #for i in idxs_to_plot:
    #    plt.plot(fbank[i]);
    #plt.legend(labels=[f'{i+1}' for i in idxs_to_plot]);
    #plt.title('Librosa Chroma Filters');
    #plt.xlabel('Sample Nb')
    #plt.ylabel('Magnitude Value')
    #plt.savefig('librosa_filters.png')

    
    #plt.figure()
    #librosa.display.specshow(fbank, x_axis='linear')
    #plt.ylabel('Chroma filter')
    #plt.title('Chroma filter bank')
    #plt.colorbar()
    #plt.tight_layout()
    
    
    return fbank



def create_fft_kernels(filter_length=512, nb_freq_pts=258):
    M_cos = numpy.eye(filter_length)
    M_sin = numpy.eye(filter_length)
    for k in range(0,filter_length-1):
        for n in range(filter_length):
            M_cos[k,n] = numpy.cos(float(n*k*2)*(numpy.pi)/float(filter_length))
            M_sin[k,n] = numpy.sin(float(n*k*2)*(numpy.pi)/float(filter_length))
                
    return M_cos, M_sin
