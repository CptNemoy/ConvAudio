import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from create_datasets import ComplexNumbersDataset
from torchsummary import summary
import librosa
from scipy.signal import chirp, spectrogram
from scipy.signal import sweep_poly
from real_signal_utils import create_fft_kernels
from real_signal_utils import create_inv_fft_kernels
device = 'cpu'


def evaluate_DFT_IDFT(model, NFFT, hop_length, eval_filename):
    # evaluate model
    model.eval()
    #audio, sr = librosa.load('got_s2e9_cake.wav')
    audio_or, sr = librosa.load('abba.wav')
    duration = librosa.get_duration(y=audio_or, sr=sr)

    np_rfft = np.fft.rfft(audio_or, NFFT)

    librosa_stft = librosa.stft(audio_or, n_fft=NFFT, hop_length=int(hop_length))

    # convert to tensor to call model
    audio = torch.FloatTensor(audio_or)
    audio = audio.unsqueeze(0)
    audio = audio.to(device)

    output_eval, time_signal_eval = model(audio)
    print(f' output_eval = {output_eval.shape}, time_signal from model {time_signal_eval.shape}')
    nb_freq_pts = output_eval.shape[0] // 2
    
    nb_frames = time_signal_eval.shape[0]
    len_frame = time_signal_eval.shape[1]
    t_signal =time_signal_eval.cpu().data.numpy()
    print(f'shape of t  signal = {t_signal.shape}')
    
    audio_or_fr = np.resize(audio_or, (nb_frames, len_frame))
    len_audio = len(audio_or)
    time_signal = np.resize(t_signal,nb_frames* len_frame )
    len_time_signal = len(time_signal)
    
    min_len = min(len_audio, len_time_signal)

    print(f' audio_or {np.shape(audio_or)}, {len_audio}  time_signal = {len_time_signal}')
    filename_time_signal = eval_filename + '_time_signal.png'
    filename_spectrum = eval_filename + '_spectrum.png'

    # Plot reconstructed time signal
    plt.figure(figsize=(6, 10));
    plt.subplot(2, 1, 1);
    plt.title('Time Signal - From Model')
    plt.ylabel('Signal Value');
    plt.xlabel('Time (samples)');
    plt.plot(time_signal / float(model.nb_freq_pts))
    plt.subplot(2, 1, 2);
    plt.title('Time Signal (Original)')
    plt.plot(audio_or)
    plt.ylabel('Signal Value');
    plt.xlabel('Time (samples)');
    plt.savefig(filename_time_signal)
    
    plt.figure(figsize=(6, 10));
    plt.plot(time_signal[10000:100100] / float(model.nb_freq_pts))
    plt.plot(audio_or[10000:100100] )
    plt.legend(labels=['time_signa', 'audio_or']);
    
    reconst_err = np.mean( ((time_signal[1:min_len] / float(model.nb_freq_pts)) - audio_or[1:min_len]) ** 2)
    print(f' reconstruction error = {reconst_err}') # on order of 1e-17
    
    # plot a spectrogram based on the dft output
    power_frames_eval = output_eval.T.squeeze()
    power_frames_eval = (output_eval[:nb_freq_pts,:] ** 2) + (output_eval[nb_freq_pts:,:] ** 2)
    print(f'power_frames_eval {power_frames_eval}')
    p_frames = power_frames_eval.cpu().data.numpy()

    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    plt.imshow(10*np.log10(1.0+p_frames), aspect='auto', origin='lower')
    plt.title('Model - Power Spectrum of audio segment');
    plt.ylabel('Frequency Index');
    plt.xlabel('Time (frame nb)');
    plt.colorbar();
    plt.subplot(1, 2, 2);
    plt.imshow(10*np.log10(1.0+abs(librosa_stft) **2), aspect='auto', origin='lower')
    #librosa.display.specshow(20*np.log10(1+MelgramFrames), sr=sr, x_axis='time', y_axis='mel')
    plt.title('Librosa-STFT-based Spectrum of audio seg');
    plt.ylabel('Frequency Index');
    plt.xlabel('Time (frame nb)');
    plt.colorbar();
    plt.savefig(filename_spectrum)
    
    return
    

def plot_initial_kernels(model, NFFT, hop_length, sr, filename_kernels):
    #
    filename_dft_kernels_all = filename_kernels + '_all_dft_kernels.png'
    filename_idft_kernels_all = filename_kernels + '_all_idft_kernels.png'
    filename_idft_kernels_select = filename_kernels + '_select_idft_kernels.png'
    filename_dft_kernels_select = filename_kernels + '_select_dft_kernels.png'

    plt.figure(figsize=(10.5, 4));
    plt.subplot(1, 2, 1);
    real_kernels = model.conv1r.weight.data
    real_kernels = real_kernels.squeeze()
    r_kernels = np.array(real_kernels)
    
    # 3D plot of DFT Kernels
    print(f'size of r_kernels = {r_kernels.shape}')
    plt.imshow(r_kernels, aspect='auto', origin='lower', cmap='seismic')
    plt.ylabel('Kernel Filter');
    plt.xlabel('Sample Index');
    plt.colorbar();
    plt.title('Initial DFT Kernel Coeffs (Real)');
    plt.subplot(1, 2, 2);
    imag_kernels = model.conv1i.weight.data
    imag_kernels = imag_kernels.squeeze()
    i_kernels = np.array(imag_kernels)
    plt.imshow(i_kernels, aspect='auto', origin='lower', cmap='seismic')
    plt.ylabel('Kernel Filter');
    plt.xlabel('Sample Index');
    plt.colorbar();
    plt.title('Initial DFT Kernel Coeffs (Imag)');
    plt.savefig(filename_dft_kernels_all)

    # plot a selected few
    plt.figure(figsize=(4, 8));
    plt.subplot(2, 1, 1);
    
    idxs_to_plot = [0, 6, 12]
    for i in idxs_to_plot:
        plt.plot(r_kernels[i,:]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Initial DFT Kernel Coeffs (Real)');
    plt.ylabel('Amplitude Value');

    plt.subplot(2, 1, 2);
    for i in idxs_to_plot:
        plt.plot(i_kernels[i,:]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Initial DFT Kernel Coeffs (Imag)');
    plt.ylabel('Amplitude Value');
    plt.xlabel('Sample index');
    plt.savefig(filename_dft_kernels_select)

    # Compute the difference between actual and analytical
    cos_kernel, sin_kernel = create_fft_kernels(filter_length=NFFT, nb_freq_pts=(NFFT//2)+1)

    cos_err = np.mean((r_kernels - cos_kernel)**2.0 )
    sin_err = np.mean((i_kernels + sin_kernel)**2.0 )
    
    print(f' cos err = {cos_err}, sin err = {sin_err}')
    print(f' size of cos_kernel = {cos_kernel.shape},  r_kernels = {r_kernels.shape}')
    # inverse kernels
    plt.figure(figsize=(10.5, 4));
    plt.subplot(1, 2, 1);
    real_kernels = model.inv_conv1r.weight.data
    real_kernels = real_kernels.squeeze()
    r_kernels = np.array(real_kernels)

    # 3D plot of I-DFT Kernels
    plt.imshow(r_kernels, aspect='auto', origin='lower', cmap='seismic')
    plt.ylabel('Kernel Filter');
    plt.xlabel('Sample Index');
    plt.colorbar();
    plt.title('Initial Inv-DFT Kernel Coeffs (Real)');
    plt.subplot(1, 2, 2);
    imag_kernels = model.inv_conv1i.weight.data
    imag_kernels = imag_kernels.squeeze()
    i_kernels = np.array(imag_kernels)
    plt.imshow(i_kernels, aspect='auto', origin='lower', cmap='seismic')
    plt.ylabel('Kernel Filter');
    plt.xlabel('Sample Index');
    plt.colorbar();
    plt.title('Initial Inv-DFT Kernel Coeffs (Imag)');
    plt.savefig(filename_idft_kernels_all)
    
    # Plot a selected few
    plt.figure(figsize=(4, 9));
    plt.subplot(2, 1, 1);
    
    idxs_to_plot = [1, 6, 12]
    for i in idxs_to_plot:
        plt.plot(r_kernels[i,:]);
    plt.legend(labels=[f'{i+1}' for i in idxs_to_plot]);
    plt.title('Initial Inv-DFT Kernel Coeffs (Real)');
    plt.ylabel('Amplitude Value');

    plt.subplot(2, 1, 2);
    for i in idxs_to_plot:
        plt.plot(i_kernels[i,:]);
    plt.legend(labels=[f'{i+1}' for i in idxs_to_plot]);
    plt.title('Initial Inv-DFT Kernel Coeffs (Imag)');
    plt.ylabel('Amplitude Value');
    plt.xlabel('Sample index');
    plt.savefig(filename_idft_kernels_select)


def plot_final_kernels(model, NFFT, hop_length, sr, filename_kernels, ini_r_kernels, ini_i_kernels, ini_inv_r_kernels, ini_inv_i_kernels):
    filename_dft_kernels_all = filename_kernels + '_all_dft_kernels.png'
    filename_dft_kernels_select = filename_kernels + '_select_dft_kernels.png'
    filename_idft_kernels_all = filename_kernels + '_all_idft_kernels.png'
    filename_idft_kernels_select = filename_kernels + '_select_idft_kernels.png'

    # Plot the Final values of hte kernels in Conv1d
    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    real_kernels = model.conv1r.weight.data
    real_kernels = real_kernels.squeeze()
    r_kernels = np.array(real_kernels)
    plt.imshow(r_kernels, aspect='auto', origin='lower', cmap='seismic')
    plt.ylabel('Kernel Filter');
    plt.xlabel('Sample Index');
    plt.colorbar();
    plt.title('Final DFT Kernel Coeffs (Real).');
    plt.subplot(1, 2, 2);
    imag_kernels = model.conv1i.weight.data
    imag_kernels = imag_kernels.squeeze()
    i_kernels = np.array(imag_kernels)
    plt.imshow(i_kernels, aspect='auto', origin='lower', cmap='seismic')
    plt.ylabel('Kernel Filter');
    plt.xlabel('Sample Index');
    plt.colorbar();
    plt.title('Final DFT Kernel Coeffs (Imag).');
    plt.savefig(filename_dft_kernels_all)

    
    plt.figure(figsize=(4, 9));
    plt.subplot(2, 1, 1);
    
    idxs_to_plot = [0, 6, 12]
    for i in idxs_to_plot:
        plt.plot(r_kernels[i,:]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Final Kernel Coefficients (Real)');
    plt.ylabel('Amplitude Value');

    plt.subplot(2, 1, 2);
    for i in idxs_to_plot:
        plt.plot(i_kernels[i,:]);
    plt.legend(labels=[f'{i}' for i in idxs_to_plot]);
    plt.title('Final Kernel Coefficients (Imag)');
    plt.ylabel('Amplitude Value');
    plt.xlabel('Sample index');
    plt.savefig(filename_dft_kernels_select)

    cos_kernel, sin_kernel = create_fft_kernels(filter_length=NFFT, nb_freq_pts=(NFFT//2)+1)

    cos_err = np.mean((r_kernels - cos_kernel)**2.0 )
    sin_err = np.mean((i_kernels + sin_kernel)**2.0 )
    
    print(f' Analytical - final : cos err = {cos_err}, sin err = {sin_err}')


    print(f'size of ini_r_kernels = {ini_r_kernels.shape}')
    print(f'size of r_kernels = {r_kernels.shape}')
    print(f'size of ini_i_kernels = {ini_i_kernels.shape}')

    if_cos_err = torch.mean((ini_r_kernels - real_kernels)**2.0)
    if_sin_err = torch.mean((ini_i_kernels - imag_kernels)**2.0)

    print(f' Initial - final :   cos err = {if_cos_err}, sin err = {if_sin_err}')

    
    # inverse kernels 3D plots
    plt.figure(figsize=(10, 4));
    plt.subplot(1, 2, 1);
    real_kernels = model.inv_conv1r.weight.data
    real_kernels = real_kernels.squeeze()
    r_kernels = np.array(real_kernels)
    plt.imshow(r_kernels, aspect='auto', origin='lower', cmap='seismic')
    plt.ylabel('Kernel Filter');
    plt.xlabel('Sample Index');
    plt.colorbar();
    plt.title('Final Inv-DFT Kernel Coeffs (Real)');
    plt.subplot(1, 2, 2);
    imag_kernels = model.inv_conv1i.weight.data
    imag_kernels = imag_kernels.squeeze()
    i_kernels = np.array(imag_kernels)
    plt.imshow(i_kernels, aspect='auto', origin='lower', cmap='seismic')
    plt.ylabel('Kernel Filter');
    plt.xlabel('Sample Index');
    plt.colorbar();
    plt.title('Final Inv-DFT Kernel Coeffs (Imag)');
    plt.savefig(filename_idft_kernels_all)

    if_inv_cos_err = torch.mean((ini_inv_r_kernels - real_kernels)**2.0)
    if_inv_sin_err = torch.mean((ini_inv_i_kernels - imag_kernels)**2.0)
    print(f' Initial - final (Inverse) :   cos err = {if_inv_cos_err}, sin err = {if_inv_sin_err}')


    # Plot selected few
    plt.figure(figsize=(4, 9));
    plt.subplot(2, 1, 1);
    
    idxs_to_plot = [1, 6, 12]
    for i in idxs_to_plot:
        plt.plot(r_kernels[i,:]);
    plt.legend(labels=[f'{i+1}' for i in idxs_to_plot]);
    plt.title('Final Inverse Kernel Coefficients (Real)');
    plt.ylabel('Amplitude Value');

    plt.subplot(2, 1, 2);
    for i in idxs_to_plot:
        plt.plot(i_kernels[i,:]);
    plt.legend(labels=[f'{i+1}' for i in idxs_to_plot]);
    plt.title('Final Inverse Kernel Coefficients (Imag)');
    plt.ylabel('Amplitude Value');
    plt.xlabel('Sample index');
    plt.savefig(filename_idft_kernels_select)

    # Compute error with analytical ones
    i_cos_kernel, i_sin_kernel = create_inv_fft_kernels(filter_length=(NFFT//2)+1, nb_time_pts=NFFT)

    cos_err = np.mean((r_kernels - i_cos_kernel)**2.0 )
    sin_err = np.mean((i_kernels - i_sin_kernel)**2.0 )
    
    print(f' Analytical (Inverse) - final : cos err = {cos_err}, sin err = {sin_err}')
