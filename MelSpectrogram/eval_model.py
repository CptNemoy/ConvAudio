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


# Read an audio file and run it thru the model. Plot the output and compare to doing a manual Mel-Spectrogram
#
def evaluate_melgram(model, NFFT, hop_length, fbank, eval_filename):
    filename_spectrum = eval_filename + '_spectrum.png'
    filename_spectrum_model_only = eval_filename + '_spectrum_model_only.png'
    filename_spectrum_vs_librosa = eval_filename + '_spectrum_model_vs_librosa.png'

    # evaluate model
    model.eval()
    
    # Read audio
    #audio, sr = librosa.load('got_s2e9_cake.wav')
    audio, sr = librosa.load('salsa.wav')
    #audio, sr = librosa.load('oriental_22050.wav')
    #audio, sr = librosa.load('abba.wav')
    duration = librosa.get_duration(y=audio, sr=sr)
    print(f'Size of audio = {audio.size}, sr = {sr}. duration = {duration}')

    # STFT-based -manual- melgram : take a DFT and apply Mel filters to it
    #
    librosa_stft = librosa.stft(audio, n_fft=NFFT, hop_length=int(hop_length), window='boxcar')
    pow_frames = (( (librosa_stft.real) ** 2 + (librosa_stft.imag) ** 2)    )   # Power Spectrum
    MelgramFrames = np.dot(pow_frames.T, fbank.T)
    MelgramFrames = MelgramFrames.T
    MelgramFrames = librosa.util.normalize(MelgramFrames, norm=4.0, axis=1)      # this normalization is not really needed and it does change the results

    # use librosa directly on the time signal  (This will give a very different result because of the filters used are different, in addition to normalization)
    librosa_mel = librosa.feature.melspectrogram(y=audio, sr=sr, S=None, n_fft=NFFT, hop_length=int(hop_length), power=4.0, n_mels=64, window='boxcar' )

    # convert audio array to tensor to call model
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)
    audio = audio.to(device)
    # Call model
    output_eval, mel_out_eval, power_frames_eval = model(audio)

    # Get the Power and the Mel Frames
    power_frames_eval = power_frames_eval.T.squeeze()
    p_frames = power_frames_eval.cpu().data.numpy()
    mel_out_eval = mel_out_eval.T.squeeze()
    m_frames = mel_out_eval.cpu().data.numpy()
    m_frames = librosa.util.normalize(m_frames, norm=4.0, axis=1)     # this normalization is not really needed and it does change the results
    
    
    # Plot Melgrams : from model and STFT-based
    plt.figure(figsize=(9, 7));
    plt.subplot(2, 1, 1);
    plt.imshow(10*np.log10(1.0+m_frames), aspect='auto', origin='lower')
    plt.title('Model - Melgram of audio segment');
    plt.colorbar();
    plt.ylabel('Filter Index');
    plt.subplot(2, 1, 2);
    plt.imshow(10*np.log10(1.0+MelgramFrames), aspect='auto', origin='lower')
    plt.title('Librosa-STFT-based Melgram of audio seg');
    plt.ylabel('Filter Index');
    plt.xlabel('Frame Nb.');
    plt.colorbar();
    plt.savefig(filename_spectrum)
    
    # Compare the Model to Librosa's native function (just for reference)
    plt.figure(figsize=(9, 7));
    plt.subplot(2, 1, 1);
    plt.imshow(10*np.log10(1.0+m_frames), aspect='auto', origin='lower')
    plt.title('Model - Melgram of audio segment');
    plt.colorbar();
    plt.ylabel('Filter Index');
    plt.subplot(2, 1, 2);
    plt.imshow(10*np.log10(1.0+librosa_mel), aspect='auto', origin='lower')
    plt.title('Librosa - Melgram of audio segment');
    plt.xlabel('Frame nb')
    plt.ylabel('Mel Filter nb')
    plt.colorbar();
    plt.savefig(filename_spectrum_vs_librosa)

    len = mel_out_eval.shape[0]
    
    mse = np.mean(( MelgramFrames[:,:len] - m_frames[:,len])**2.0)
    
    print(f' MSE of Melgram = {mse}')
    
    
# Plot the initial kernels of the DFT and Mel filters
#       compute some errors
#
def plot_initial_kernels(model, NFFT, hop_length, sr, fbank, n_mels, filename_kernels, filename_mels):
    #
    filename_dft_kernels_all = filename_kernels + '_all_dft_kernels.png'
    filename_dft_kernels_select = filename_kernels + '_select_mel_kernels.png'

    # Plot the DFT kernels
    #   Real and Imag
    #
    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    real_kernels = model.conv1r.weight.data
    real_kernels = real_kernels.squeeze()
    r_kernels = np.array(real_kernels)
    plt.imshow(r_kernels, aspect='auto', origin='lower', cmap='seismic')
    plt.ylabel('Kernel Filter');
    plt.xlabel('Sample Index');
    plt.colorbar();
    plt.title('Initial DFT Kernel Coefficients (Real)');
    plt.subplot(1, 2, 2);
    imag_kernels = model.conv1i.weight.data
    imag_kernels = imag_kernels.squeeze()
    i_kernels = np.array(imag_kernels)
    plt.imshow(i_kernels, aspect='auto', origin='lower', cmap='seismic')
    plt.ylabel('Kernel Filter');
    plt.xlabel('Sample Index');
    plt.colorbar();
    plt.title('Initial DFT Kernel Coefficients (Imag)');
    plt.savefig(filename_dft_kernels_all)

    # plot the initial values of the Mel filters in the conv bank.
    #
    plt.figure(figsize=(5, 4));
    init_mels_kernels = model.conv1d_mels.weight.data
    init_mels_kernels = init_mels_kernels.squeeze()
    i_m_kernels = np.array(init_mels_kernels)
    librosa.display.specshow(i_m_kernels, hop_length=hop_length, x_axis='frames', y_axis='frames', cmap='inferno');
    plt.ylabel('Mel filter Index');
    plt.xlabel('Sample nb');
    plt.title('Initial Mel Coefficients');
    plt.colorbar();
    plt.savefig(filename_mels)
    
    # Error between starting values of Mel filters and analytical solution
    m_err = np.mean( np.abs( i_m_kernels - fbank) )
    print(f' size of i_m_kernels = {i_m_kernels.shape}.  size of fbank = {fbank.shape}.  m_err = {m_err}')

    # Plot a selected few filters
    plt.figure(figsize=(6, 6));
    idxs_to_plot = [2, 11, 21, 33, 44, n_mels-9, n_mels-3]

    for i in idxs_to_plot:
        plt.plot(i_m_kernels[i]);
    plt.legend(labels=[f'{i+1}' for i in idxs_to_plot]);
    plt.title('Initial Mel filters');
    plt.xlabel('Sample Index')
    plt.ylabel('Magnitude Value')
    plt.savefig(filename_dft_kernels_select)


# Plot the final kernels of the DFT and Mel filters
#       compute some errors
#
def plot_final_kernels(model, NFFT, hop_length, sr, fbank, n_mels, filename_kernels, filename_mels, initial_mels):
    #
    filename_dft_kernels_all = filename_kernels + '_all_dft_kernels.png'
    filename_dft_kernels_select = filename_kernels + '_select_mel_kernels.png'
    
    # Plot the Final values of the DFT kernels in Conv1d
    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    real_kernels = model.conv1r.weight.data
    real_kernels = real_kernels.squeeze()
    r_kernels = np.array(real_kernels)
    plt.imshow(r_kernels, aspect='auto', origin='lower', cmap='seismic')
    #librosa.display.specshow(r_kernels, hop_length=hop_length, x_axis='frames', y_axis='frames');
    plt.ylabel('Kernel Filter');
    plt.xlabel('Sample Index');
    plt.colorbar();
    plt.title('Final DFT Kernel Coeff (Real) . After Training');
    plt.subplot(1, 2, 2);
    imag_kernels = model.conv1i.weight.data
    imag_kernels = imag_kernels.squeeze()
    i_kernels = np.array(imag_kernels)
    plt.imshow(i_kernels, aspect='auto', origin='lower', cmap='seismic')
    #librosa.display.specshow(i_kernels, hop_length=hop_length, x_axis='frames', y_axis='frames');
    plt.ylabel('Kernel Filter');
    plt.xlabel('Sample Index');
    plt.colorbar();
    plt.title('Final DFT Kernel Coeff (Imag) . After Training');
    plt.savefig(filename_dft_kernels_all)

    # Plot All the  Mel kernels
    plt.figure(figsize=(5, 4));
    mel_kernels = model.conv1d_mels.weight.data
    mel_kernels = mel_kernels.squeeze()
    m_kernels = np.array(mel_kernels)
    librosa.display.specshow(m_kernels, hop_length=hop_length, x_axis='frames', y_axis='frames',cmap='inferno');
    plt.ylabel('Mel filter Index');
    plt.xlabel('Sample nb');
    plt.title('Final Mel Coefficients (after training)');
    plt.colorbar();
    plt.savefig(filename_mels)

    # Error between final and analytical solution
    m_err = np.mean( ( m_kernels - fbank)**2.0 )
    print(f' size of m_kernels = {m_kernels.shape}.  size of fbank = {fbank.shape}.  m_err = {m_err}')

    # Error between final and initial solution (in case of initialized Mel)
    f_i_err = np.mean( ( m_kernels - initial_mels)**2.0 )
    print(f' I_F_err = {f_i_err}')

    # Plot a selected few filters
    plt.figure(figsize=(6, 6));
    idxs_to_plot = [2, 11, 21, 33, 44, n_mels-9, n_mels-3]
    for i in idxs_to_plot:
        plt.plot(fbank[i]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Precomputed Mel filters');


    plt.figure(figsize=(6, 6));
    idxs_to_plot = [2, 11, 21, 33, 44, n_mels-9, n_mels-3]
    for i in idxs_to_plot:
        plt.plot(m_kernels[i]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Final Mel filters');
    plt.xlabel('Sample Index')
    plt.ylabel('Magnitude Value')
    plt.savefig(filename_dft_kernels_select)
