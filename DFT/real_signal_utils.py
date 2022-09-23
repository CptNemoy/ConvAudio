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



def create_inv_fft_kernels(filter_length=514, nb_time_pts=1024):
    M_cos = numpy.zeros((nb_time_pts,filter_length), dtype=float)
    M_sin = numpy.zeros((nb_time_pts,filter_length), dtype=float)
    for n in range(nb_time_pts):
        for k in range(filter_length):
            M_cos[n,k] = numpy.cos(float(n*k*2)*(numpy.pi)/float(nb_time_pts))
            M_sin[n,k] = numpy.sin(float(n*k*2)*(numpy.pi)/float(nb_time_pts))

    return M_cos, M_sin



def create_fft_kernels(filter_length=1024, nb_freq_pts=513):
    M_cos = numpy.zeros((nb_freq_pts,filter_length), dtype=float)
    M_sin = numpy.zeros((nb_freq_pts,filter_length), dtype=float)
    for k in range(nb_freq_pts):
        for n in range(filter_length):
            M_cos[k,n] = numpy.cos(float(n*k*2)*(numpy.pi)/float(filter_length))
            M_sin[k,n] = numpy.sin(float(n*k*2)*(numpy.pi)/float(filter_length))
    return M_cos, M_sin
