import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from real_signal_utils import create_fft_kernels
from real_signal_utils import create_inv_fft_kernels

class ConvDFT(nn.Module):
    def __init__(self, N=32, hop_length=16, win_length=None,window='hann', sr=16000, init_dft_kernels=False, init_idft_kernels=False):
        super(ConvDFT, self).__init__()
        self.N = N
        self.hop_length = hop_length
        self.win_length = win_length if win_length else N
        self.window = window
        self.initialize_dft_kernels = init_dft_kernels
        self.initialize_idft_kernels = init_idft_kernels
        self.sampling_rate = sr
        self.nb_freq_pts = int(self.N/2)+1
        #
        # outputs = nb of filters = N (N/2 + 1 real, N/2 + 1Imag)
        # kernel size = N
        self.conv1r = nn.Conv1d(in_channels=1, out_channels=self.nb_freq_pts, kernel_size=self.N, stride=int(hop_length))
        self.conv1i = nn.Conv1d(in_channels=1, out_channels=self.nb_freq_pts, kernel_size=self.N, stride=int(hop_length))

        self.inv_conv1r = nn.Conv1d(in_channels=1, out_channels=self.N, kernel_size=self.nb_freq_pts, stride=int(hop_length))
        self.inv_conv1i = nn.Conv1d(in_channels=1, out_channels=self.N, kernel_size=self.nb_freq_pts, stride=int(hop_length))

        # Get initial DFT kernels
        cos_kernel, sin_kernel = create_fft_kernels(filter_length=N, nb_freq_pts=self.nb_freq_pts)
        
        # convert to Torch tensor . This matrix is now the stack of kernels (note : size (N/2+1) x N)
        cos_kernel_dft = torch.FloatTensor(cos_kernel[:, None, :])
        sin_kernel_dft = torch.FloatTensor(-sin_kernel[:, None, :])
        
        self.register_buffer('cos_kernel_dft', cos_kernel_dft.float())
        self.register_buffer('sin_kernel_dft', sin_kernel_dft.float())

        print(f' size of dft = {cos_kernel_dft.shape}')
        # initialize the kernels of the DFT w/ the cos and -sin kernels
        if (self.initialize_dft_kernels == True):
            print(f'initializing 1st bandk of convolutions')
            with torch.no_grad():
                self.conv1r.weight = torch.nn.Parameter(cos_kernel_dft)
                self.conv1i.weight = torch.nn.Parameter(sin_kernel_dft)

        # Get initial I-DFT kernels
        i_cos_kernel, i_sin_kernel = create_inv_fft_kernels(filter_length=self.nb_freq_pts, nb_time_pts=self.N)

        i_cos_kernel_idft = torch.FloatTensor(( i_cos_kernel[:, None, :]))
        i_sin_kernel_idft = torch.FloatTensor(( i_sin_kernel[:, None, :]))

        # initialize the kernels of the IDFT w/ the cos and -sin kernels
        if (self.initialize_idft_kernels == True):
            print(f'initializing 1st bandk of convolutions')
            with torch.no_grad():
                self.inv_conv1r.weight = ( torch.nn.Parameter(i_cos_kernel_idft))
                self.inv_conv1i.weight = ( torch.nn.Parameter(i_sin_kernel_idft))



    def forward(self, x):
        # Dense layer only for now
        # x = self.fc1(x)
        num_batches = x.shape[0]
        num_samples = x.shape[1]
        
        # put input in format [1,1,L]    where L is the incoming samples
        # The shape of the internal kernels is [F, 1, N]     where F is the nb of filter, and N is the filter length
        #
        x = torch.reshape(x, (num_batches, 1, num_samples) )   # last dimension added to accomodate the conv2d trick
        
        y_real = self.conv1r(x)
        y_imag = self.conv1i(x)
        y_real = y_real.squeeze(0)                  # y is in shape [F, b] where b is the nb of outputs. if L = N, then b = 1
        y_imag = y_imag.squeeze(0)                  # y is in shape [F, b] where b is the nb of outputs. if L = N, then b = 1
 

        y = torch.cat([y_real, y_imag], dim=0)      # [y_real, y_imag ] stacked vertically

        nb_of_frames = y_real.shape[1]
        
        y_real = y_real.T.unsqueeze(1)    # reorganize in proper shape [nb_frames, 1, nb_freq_pts]  eg [2,1,513]
        y_imag = y_imag.T.unsqueeze(1)    # reorganize in proper shape

        real_real = self.inv_conv1r(y_real)
        imag_imag = self.inv_conv1i(y_imag)
        
        time_signal = real_real - imag_imag
        time_signal = time_signal.squeeze()  # / float(self.nb_freq_pts)       # shape [b, N] where b is the nb of output framess. N=1024 (same length as the input)
        return y, time_signal
        
