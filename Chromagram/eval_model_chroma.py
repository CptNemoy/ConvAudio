import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from create_datasets import ComplexNumbersDataset
from torchsummary import summary
import librosa
device = 'cpu'


def evaluate_chroma(model, NFFT, hop_length, fbank, eval_filename, n_chroma, librosa_filename):
    filename_spectrum = eval_filename + '_spectrum.png'
    filename_spectrum_model_only = eval_filename + '_spectrum_model_only.png'

    # evaluate model
    model.eval()
    
    # read an audio file
    audio, sr = librosa.load('got_s2e9_cake.wav')
    #audio, sr = librosa.load('abba.wav')
    duration = librosa.get_duration(y=audio, sr=sr)

    # compute chromagram from STFT and filterbank
    librosa_stft = librosa.stft(audio, n_fft=NFFT, hop_length=int(hop_length), window='boxcar')
    pow_frames = (( (librosa_stft.real) ** 2 + (librosa_stft.imag) ** 2)    )   # Power Spectrum
    ChromaFrames = np.dot(pow_frames.T, fbank.T)
    ChromaFrames = ChromaFrames.T
    ChromaFrames = librosa.util.normalize(ChromaFrames, norm=4.0, axis=1)

    # call librosa chroma  on the power spectrum
    chroma_power = librosa.feature.chroma_stft(y=None, sr=sr, S=pow_frames, n_fft=NFFT, hop_length=hop_length, tuning=0.0, n_chroma=n_chroma, norm=4.0)

    # call librosa chroma directly on the original signal
    chroma_signal = librosa.feature.chroma_stft(y=audio, sr=sr, S=None, n_fft=NFFT, hop_length=hop_length, tuning=0.0, n_chroma=n_chroma, norm=4.0, window='boxcar' )

    # convert to tensor to call model
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)
    audio = audio.to(device)

    output_eval, chroma_out_eval, power_frames_eval = model(audio)

    # convert to numpy arrays and normalize
    power_frames_eval = power_frames_eval.T.squeeze()
    p_frames = power_frames_eval.cpu().data.numpy()

    chroma_out_eval = chroma_out_eval.T.squeeze()
    c_frames = chroma_out_eval.cpu().data.numpy()
    c_frames = librosa.util.normalize(c_frames, norm=4.0, axis=0)
    
    # Plot the 3D Chromagram: Model vs. Librosa
    plt.figure(figsize=(9, 7));
    plt.subplot(211);
    plt.imshow(10*np.log10(1.0+c_frames), aspect='auto', origin='lower')
    plt.title('Model - Chromagram of Audio Seg');
    plt.ylabel('Chroma Filter Nb');
    #plt.xlabel('Time (Frames)');
    plt.colorbar();
    plt.subplot(212);
    plt.imshow(10*np.log10(1.0+chroma_signal), aspect='auto', origin='lower')
    plt.title('Librosa Chromagram of Audio Seg');
    plt.ylabel('Chroma Filter Nb');
    plt.xlabel('Time (Frames)');
    plt.colorbar();
    plt.savefig(filename_spectrum)
    
    # compute the mse
    len = chroma_out_eval.shape[0]
    mse = np.mean(( c_frames[:,:len] - chroma_signal[:,len])**2.0)
    
    print(f' MSE of Chroma = {mse}')

    mse = np.mean(    ( c_frames[:,:len] - chroma_signal[:,len])**2.0 / ( chroma_signal[:,:len] )**2.0)
    print(f' Relative MSE of Chroma = {mse}')

    # plot the librosa chroma
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1);
    plt.imshow((1.0+chroma_power), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Librosa Chroma gram - Power')
    # plot the librosa chroma
    plt.subplot(1, 2, 2);
    plt.imshow((1.0+chroma_signal), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Librosa Chroma gram - Signal')
    plt.savefig(librosa_filename)

    plt.show()


def plot_initial_kernels(model, NFFT, hop_length, sr, fbank, filename_kernels, filename_chroma, n_chroma):
    #
    filename_dft_kernels_all = filename_kernels + '_all_dft_kernels.png'
    filename_dft_kernels_select = filename_kernels + '_select_chroma_kernels.png'
    #
    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    real_kernels = model.conv1r.weight.data
    real_kernels = real_kernels.squeeze()
    r_kernels = np.array(real_kernels)
    librosa.display.specshow(r_kernels, hop_length=hop_length, x_axis='frames', y_axis='frames');
    plt.ylabel('Kernel filter');
    plt.xlabel('Sample Nb');
    plt.colorbar();
    plt.title('Initial Kernel Coefficients (Real)');
    plt.subplot(1, 2, 2);
    imag_kernels = model.conv1i.weight.data
    imag_kernels = imag_kernels.squeeze()
    i_kernels = np.array(imag_kernels)
    librosa.display.specshow(i_kernels, hop_length=hop_length, x_axis='frames', y_axis='frames');
    plt.ylabel('Initial Kernel filter');
    plt.xlabel('Sample Nb');
    plt.colorbar();
    plt.title('Initial Kernel Coefficients (Imag)');
    plt.savefig(filename_dft_kernels_all)

    # plot the initial values of the Chroma filters in the conv bank.
    plt.figure(figsize=(5, 4));

    init_chroma_kernels = model.conv1d_chroma.weight.data
    init_chroma_kernels = init_chroma_kernels.squeeze()
    i_ch_kernels = np.array(init_chroma_kernels)
    print(f'i_ch_kernels size = {i_ch_kernels.shape}. ')
    librosa.display.specshow(i_ch_kernels, hop_length=hop_length, x_axis='frames', y_axis='frames', cmap='inferno');
    plt.ylabel('Chroma filter Index');
    plt.xlabel('Sample Nb');
    plt.title('Initial Chroma Coefficients');
    plt.colorbar();
    plt.savefig(filename_chroma)

    plt.figure(figsize=(6, 4));
    idxs_to_plot = [ 11, n_chroma-9, n_chroma-3]

    for i in idxs_to_plot:
        plt.plot(i_ch_kernels[i]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Initial Chroma filters');
    plt.xlabel('Sample Nb')
    plt.ylabel('Magnitude Value')
    plt.savefig(filename_dft_kernels_select)
    m_err = np.mean( ( i_ch_kernels - fbank)**2.0 )
    print(f' Error between analytical & actual :  m_err = {m_err}')



def plot_final_kernels(model, NFFT, hop_length, sr, fbank, n_chroma,filename_kernels, filename_chroma, filename_chroma_filters, initial_c_kernels):
    #
    filename_dft_kernels_all = filename_kernels + '_all_dft_kernels.png'
    filename_dft_kernels_select = filename_kernels + '_select_chroma_kernels.png'
    
    # Plot the DFT kernels
    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    real_kernels = model.conv1r.weight.data
    real_kernels = real_kernels.squeeze()
    r_kernels = np.array(real_kernels)
    librosa.display.specshow(r_kernels, hop_length=hop_length, x_axis='frames', y_axis='frames');
    plt.ylabel('Kernel filter');
    plt.xlabel('filter sample');
    plt.colorbar();
    plt.title('Final Kernel Coefficients (Real)');
    plt.subplot(1, 2, 2);
    imag_kernels = model.conv1i.weight.data
    imag_kernels = imag_kernels.squeeze()
    i_kernels = np.array(imag_kernels)
    librosa.display.specshow(i_kernels, hop_length=hop_length, x_axis='frames', y_axis='frames');
    plt.ylabel('Final Kernel filter');
    plt.xlabel('Sample Nb');
    plt.colorbar();
    plt.title('Final Kernel Coefficients (Imag)');
    plt.savefig(filename_dft_kernels_all)

    # plot the initial values of the Chroma filters in the conv bank.
    plt.figure(figsize=(5, 4));
    final_chroma_kernels = model.conv1d_chroma.weight.data
    final_chroma_kernels = final_chroma_kernels.squeeze()
    f_ch_kernels = np.array(final_chroma_kernels)
    print(f'f_ch_kernels size = {f_ch_kernels.shape}. ')
    librosa.display.specshow(f_ch_kernels, hop_length=hop_length, x_axis='frames', y_axis='frames', cmap='inferno');
    plt.ylabel('Chroma filter Index');
    plt.xlabel('Sample Nb');
    plt.title('Final Chroma Coefficients');
    plt.colorbar();
    plt.savefig(filename_chroma)

    # error between final kernels and analytical (Librosa filters)
    m_err = np.mean( ( f_ch_kernels - fbank)**2.0 )
    print(f'Error between Analytical and Actual :    m_err = {m_err}')

    # Error between initial and final kernels
    f_i_err = np.mean( ( f_ch_kernels - initial_c_kernels)**2.0 )
    print(f' I_F_err = {f_i_err}')


    plt.figure(figsize=(6, 4));
    idxs_to_plot = [11,  n_chroma-9, n_chroma-3]

    for i in idxs_to_plot:
        plt.plot(f_ch_kernels[i]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Final Chroma filters');
    plt.xlabel('Sample Nb')
    plt.ylabel('Magnitude Value')
    plt.savefig(filename_dft_kernels_select)
