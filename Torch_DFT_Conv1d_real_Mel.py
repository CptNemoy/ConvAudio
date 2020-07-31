import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from speech_melgram import create_mel_filters
from speech_melgram import create_fft_kernels

#from .util import window_sumsquare

class ConvDFT(nn.Module):

    def __init__(self, filter_length=32, hop_length=16, win_length=None,window='hann', n_mels=32, sr=16000):
        super(ConvDFT, self).__init__()
        self.N = filter_length
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        self.pad_amount = int(self.filter_length / 2)
        self.initialize_kernels = False
        self.n_mels = n_mels
        self.sampling_rate = sr
        #
        # 1D input
        self.nb_freq_pts = int((self.N/2)+1)
        # outputs = nb of filters = N (N/2 + 1 real, N/2 + 1Imag)
        # kernel size = N
        self.conv1r = nn.Conv1d(in_channels=1, out_channels=self.nb_freq_pts, kernel_size=self.N, stride=int(hop_length))
        self.conv1i = nn.Conv1d(in_channels=1, out_channels=self.nb_freq_pts, kernel_size=self.N, stride=int(hop_length))
        #
        # Fully connected layers (not used)
        self.fc1 = nn.Linear(in_features=(2*self.nb_freq_pts), out_features=(2*self.nb_freq_pts))
        print(f' out_channels = {2*self.nb_freq_pts}. kernel size = {self.N}' )
        self.fc_mels = nn.Linear(in_features=(self.nb_freq_pts), out_features=self.n_mels)
        
        # Bank of Convolution Filters to implement Mel Filters
        self.conv1d_mels = nn.Conv1d(in_channels=1, out_channels=self.n_mels, kernel_size=self.nb_freq_pts, stride=1)

        #
        # Weight Initialization for the conv1
        # Generate the kernels using the FFT of an identity matrix and load them
        #
        scale = self.filter_length / self.hop_length
        cos_kernel, sin_kernel = create_fft_kernels(filter_length=filter_length, nb_freq_pts=self.nb_freq_pts)
        
        # convert to Torch tensor . This matrix is now the stack of kernels
        cos_kernel = torch.FloatTensor(cos_kernel[:, None, :])
        sin_kernel = torch.FloatTensor(sin_kernel[:, None, :])

        # Get the Mel filters
        fbank = create_mel_filters(sample_rate=self.sampling_rate, n_fft=self.N, nfilt=n_mels)
        fbank = torch.FloatTensor(fbank[:, None, :])
        print(f' size of fbank inside init = {fbank.shape}')

        # initialize the kernels of the convolutions w/ the FFT kernels
        if (self.initialize_kernels == True):
            print(f'initializing 1st bandk of convolutions')
            with torch.no_grad():
                self.conv1r.weight = torch.nn.Parameter(cos_kernel)
                self.conv1i.weight = torch.nn.Parameter(sin_kernel)

        # initialize the kernels of the convolutions w/ the Mel filters
        if (self.initialize_kernels == True):
            print(f'initializing 2nd bandk of convolutions')
            with torch.no_grad():
                self.conv1d_mels.weight = torch.nn.Parameter(fbank)

        self.init_kernel = self.conv1r.weight.data
        



    def forward(self, x):
        # Dense layer only for now
        # x = self.fc1(x)
        num_batches = x.shape[0]
        num_samples = x.shape[1]
        
        # put input in format [1,1,L]    where L is the incoming samples
        # The shape of the internal kernels is [F, 1, N]     where F is the nb of filter, and N is the filter length
        #
        x = x.view(num_batches, 1, num_samples)    # last dimension added to accomodate the conv2d trick
        #print(f' size of x at input = {x.shape}')
        
        y_real = self.conv1r(x)
        y_imag = self.conv1i(x)
        #print(f'size of y ={y.shape}')
        y_real = y_real.squeeze(0)                  # y is in shape [F, b] where b is the nb of outputs. if L = N, then b = 1
        y_imag = y_imag.squeeze(0)                  # y is in shape [F, b] where b is the nb of outputs. if L = N, then b = 1
        
        y = torch.cat([y_real, y_imag], dim=0)
        #print(f'size of y ={y.shape}')
        #return y
        # Get the power of the r and imag
        power_frame =  ((y_real) ** 2 + (y_imag) ** 2)    # Power 'Spectrum'
        #print(f'size of power_frame = {power_frame.shape}')
        
        nb_of_frames = power_frame.shape[1]
        #matrix_power_frame = power_frame.view(nb_of_frames, 1, self.nb_freq_pts)    # reorganize in proper shape
        
        matrix_power_frame = power_frame.T.unsqueeze(1)                             # reorganize in proper shape
        #print(f' shape of matrix_power_frame ={matrix_power_frame.shape}')
        
        z = self.conv1d_mels(matrix_power_frame)                                   # run thru convolution banks
        #print(f' shape of z ={z.shape}')
        return y, z, power_frame                                                        # return fft, and mel output
        
