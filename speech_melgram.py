import numpy
import scipy.io.wavfile
from scipy.fftpack import dct


import torch
from torch import nn, optim
import torch.nn.functional as F

from matplotlib.cbook import (
    MatplotlibDeprecationWarning, dedent, get_label, sanitize_sequence)
from matplotlib.cbook import mplDeprecation  # deprecated
from matplotlib.rcsetup import defaultParams, validate_backend, cycler
import matplotlib.pyplot as plt
import librosa
import librosa.display


def create_mel_filters(sample_rate=16000, n_fft=512, nfilt=32):
    # Mel Filter Banks
    # ----------------
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((n_fft + 1) * hz_points / sample_rate)


    fbank = numpy.zeros((nfilt, int(numpy.floor(n_fft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
            
    return fbank



def create_fft_kernels(filter_length=512, nb_freq_pts=258):
    # create an identity matrix.
    # each row in this matrix is a vector of Length N, has only one non-zero value
    # Take the FFT of the entire matrix. Each row is the FFT of an N-length real vector with 1 non-zero elements
    fourier_basis = numpy.fft.fft(numpy.eye(filter_length))

    # keep only the first (N/2 + 1) rows
    # Split the complex rows into 'real' rows stacked on top of 'imag' rows
    #
    cos_kernel = numpy.real(fourier_basis[:nb_freq_pts, :])
    sin_kernel = numpy.imag(fourier_basis[:nb_freq_pts, :])
    #
    return cos_kernel, sin_kernel
