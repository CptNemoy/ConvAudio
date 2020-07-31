import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from create_datasets import ComplexNumbersDataset
#from Torch_DFT_ANN import PDFT
from torchsummary import summary
import librosa
device = 'cpu'


def evaluate_melgram(model, NFFT, hop_length, fbank, eval_filename):
    # evaluate model
    model.eval()

    audio, sr = librosa.load('got_s2e9_cake.wav')
    #audio, sr = librosa.load('abba.wav')
    duration = librosa.get_duration(y=audio, sr=sr)

    np_rfft = np.fft.rfft(audio, NFFT)

    librosa_stft = librosa.stft(audio, n_fft=NFFT, hop_length=int(hop_length))

    print(f'size of np_rfft = {np_rfft.shape}')
    print(f'size of librosa_stft = {librosa_stft.shape}')

    pow_frames = (( (librosa_stft.real) ** 2 + (librosa_stft.imag) ** 2)    )   # Power Spectrum
    MelgramFrames = np.dot(pow_frames.T, fbank.T)
    MelgramFrames = MelgramFrames.T
    print(f'size of melframes = {MelgramFrames.shape}')

    # convert to tensor to call model
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)
    audio = audio.to(device)


    print(f'Done.   size of audio = {audio.size()}, sr = {sr}. duration = {duration}')
    output_eval, mel_out_eval, power_frames_eval = model(audio)
    print(f' mel_out_eval = {mel_out_eval.shape}, {power_frames_eval.shape}')


    power_frames_eval = power_frames_eval.T.squeeze()
    print(f'power_frames_eval {power_frames_eval}')
    p_frames = power_frames_eval.cpu().data.numpy()
    print(f'p frames {p_frames}')
    mel_out_eval = mel_out_eval.T.squeeze()
    m_frames = mel_out_eval.cpu().data.numpy()

    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    plt.imshow(10*np.log10(1.0+m_frames), aspect='auto', origin='lower')
    plt.title('Model - Melgram of audio segment');
    plt.colorbar();
    plt.subplot(1, 2, 2);
    plt.imshow(10*np.log10(1.0+MelgramFrames), aspect='auto', origin='lower')
    #librosa.display.specshow(20*np.log10(1+MelgramFrames), sr=sr, x_axis='time', y_axis='mel')
    plt.title('Librosa-STFT-based Melgram of audio seg');
    plt.colorbar();
    plt.savefig(eval_filename)
    plt.show()
    
    
    
    


def plot_initial_kernels(model, NFFT, hop_length, sr, fbank, filename_kernels, filename_mels):
    #
    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    real_kernels = model.conv1r.weight.data
    real_kernels = real_kernels.squeeze()
    r_kernels = np.array(real_kernels)
    librosa.display.specshow(r_kernels, sr=sr, hop_length=hop_length, x_axis='linear');
    plt.ylabel('Kernel filter');
    plt.xlabel('filter sample');
    plt.colorbar();
    plt.title('Initial Kernel Coefficients (Real)');
    plt.subplot(1, 2, 2);
    imag_kernels = model.conv1i.weight.data
    imag_kernels = imag_kernels.squeeze()
    i_kernels = np.array(imag_kernels)
    librosa.display.specshow(i_kernels, sr=sr, hop_length=hop_length, x_axis='linear');
    plt.ylabel('Initial Kernel filter');
    plt.xlabel('filter sample');
    plt.colorbar();
    plt.title('Initial Kernel Coefficients (Imag)');
    plt.savefig(filename_kernels)
    plt.show()

    # plot the initial values of the Mel filters in the conv bank.
    plt.figure(figsize=(5, 4));

    init_mels_kernels = model.conv1d_mels.weight.data
    init_mels_kernels = init_mels_kernels.squeeze()
    i_m_kernels = np.array(init_mels_kernels)
    print(f'i_m_kernels size = {i_m_kernels.shape}. ')
    librosa.display.specshow(i_m_kernels, sr=sr, hop_length=hop_length, x_axis='linear');
    plt.ylabel('Mel filter Index');
    plt.title('Initial Mel Coefficients');
    plt.colorbar();
    plt.savefig(filename_mels)
    plt.show()

def plot_final_kernels(model, NFFT, hop_length, sr, fbank, filename_kernels, filename_mels):
    # Plot the Final values of hte kernels in Conv1d
    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    real_kernels = model.conv1r.weight.data
    real_kernels = real_kernels.squeeze()
    r_kernels = np.array(real_kernels)
    librosa.display.specshow(r_kernels, sr=sr, hop_length=hop_length, x_axis='linear');
    plt.ylabel('Kernel filter');
    plt.colorbar();
    plt.title('Final Kernel Coeff (Real) . After Training');
    plt.subplot(1, 2, 2);
    imag_kernels = model.conv1i.weight.data
    imag_kernels = imag_kernels.squeeze()
    i_kernels = np.array(imag_kernels)
    librosa.display.specshow(i_kernels, sr=sr, hop_length=hop_length, x_axis='linear');
    plt.ylabel('Kernel filter');
    plt.colorbar();
    plt.title('Final Kernel Coeff (Imag) . After Training');
    plt.savefig(filename_kernels)
    plt.show()

    plt.figure(figsize=(5, 4));

    mel_kernels = model.conv1d_mels.weight.data
    mel_kernels = mel_kernels.squeeze()
    m_kernels = np.array(mel_kernels)
    librosa.display.specshow(m_kernels, sr=sr, hop_length=hop_length, x_axis='linear');
    plt.ylabel('Mel filter Index');
    plt.title('Final Mel Coefficients (after training)');
    plt.colorbar();
    plt.savefig(filename_mels)
    plt.show()
