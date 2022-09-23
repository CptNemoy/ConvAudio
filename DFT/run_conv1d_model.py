import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fft import fft, rfft, irfft
import matplotlib.pyplot as plt
import torch.optim as optim
from create_datasets import ComplexNumbersDataset
from Torch_DFT_Conv1d_real import ConvDFT
from torchsummary import summary
from eval_model import plot_initial_kernels
from eval_model import plot_final_kernels
from eval_model import evaluate_DFT_IDFT
import librosa

device = 'cpu'

# The following parameters can be changed
NFFT = 1024
nb_batches = 800
batch_size = 200
sample_rate = 16000
init_dft_kernels=False       # True if we wish to initialize the kernels
init_idft_kernels=False
hop_length = NFFT         # valid values : NFFT or  NFFT// 2
# ========================================

# create a dataset of several batches of complex vectors of size N
dataset = ComplexNumbersDataset(nb_batches, batch_size, NFFT, complex=False)
dataset_validation = ComplexNumbersDataset(nb_batches, batch_size, NFFT, complex=False)

# initialize model
c_dft = ConvDFT(N=NFFT, hop_length=hop_length, win_length=None, window='hann', sr=sample_rate, init_dft_kernels=init_dft_kernels, init_idft_kernels=init_idft_kernels)
print(c_dft)

# take a copy of the kernels at initialization time, for later comparison w/ the final values
real_kernels = (c_dft.conv1r.weight.data).clone().detach()
imag_kernels = (c_dft.conv1i.weight.data).clone().detach()
real_inv_kernels = (c_dft.inv_conv1r.weight.data).clone().detach()
imag_inv_kernels = (c_dft.inv_conv1i.weight.data).clone().detach()
print(f'size of _inv_conv1r = {real_inv_kernels.shape}')

# Plot the initial kernels
plot_initial_kernels(model=c_dft, NFFT=NFFT, hop_length=NFFT, sr=sample_rate, filename_kernels='Initial_Kernels')
evaluate_DFT_IDFT(model=c_dft, NFFT=NFFT, hop_length=NFFT,  eval_filename='initial_eval_dft_idft')


# This function is called for each batch to run training on a given bank (analysis or synthesis)
# ----------------------------------------------------------------------------------------------
def train_banks(model, epoch_ind, optimizer, scheduler, loss_fct, sig, Label, batch_size, sig_validation, Label_validation, N, analysis_bank=False, synthesis_bank=False):

    if( (analysis_bank == False) & (synthesis_bank==False)):
        print(f'Nothing to train')
        return 0, 0
    if( (analysis_bank == True) & (synthesis_bank==False)):
        model.conv1r.weight.requires_grad = True
        model.conv1r.bias.requires_grad = True
        model.conv1i.weight.requires_grad = True
        model.conv1i.bias.requires_grad = True
        # synthesis bank off..
        model.inv_conv1r.weight.requires_grad = False
        model.inv_conv1r.bias.requires_grad = False
        model.inv_conv1i.weight.requires_grad = False
        model.inv_conv1i.bias.requires_grad = False
    if( (analysis_bank == False) & (synthesis_bank==True)):
        model.conv1r.weight.requires_grad = False
        model.conv1r.bias.requires_grad = False
        model.conv1i.weight.requires_grad = False
        model.conv1i.bias.requires_grad = False
        # synthesis bank ON..
        model.inv_conv1r.weight.requires_grad = True
        model.inv_conv1r.bias.requires_grad = True
        model.inv_conv1i.weight.requires_grad = True
        model.inv_conv1i.bias.requires_grad = True


    train_loss, valid_loss = [], []
    model.train()
    
    for idx in range(batch_size):
        # zero gradients
        optimizer.zero_grad()

        # get one row of input
        input_x = sig[idx, :]
        
        #Convert to tensor to call model with it
        input_x = torch.FloatTensor(input_x)
        input_x = input_x.unsqueeze(0)
        input_x = input_x.to(device)

        # run forward
        output, time_signal = model(input_x)
        output = output.squeeze()
        
        time_signal = time_signal.squeeze()
        input_x = input_x.squeeze(0)
        
        # get the desired result
        target = Label[idx, :]
        target = torch.FloatTensor(target)
        target = target.to(device)
        
        # get error
        if( (analysis_bank == True) & (synthesis_bank==False)):
            loss = loss_fct(output, target)
            train_loss.append(loss.item())
            loss.backward()
        if( (analysis_bank == False) & (synthesis_bank==True)):
            loss_time_signal = loss_fct(time_signal, float(model.nb_freq_pts) * input_x)
            train_loss.append(loss_time_signal.item())
            loss_time_signal.backward()

        # backward prop
        optimizer.step()
        
    ## evaluation part
    model.eval()
    for idx in range(batch_size):
        # get one row of input
        input_x_valid = sig_validation[idx, :]
        #
        # input_x has to be in [1, L]    shape
        input_x_valid = torch.FloatTensor(input_x_valid)
        input_x_valid = input_x_valid.unsqueeze(0)
        input_x_valid = input_x_valid.to(device)
        #
        output_valid, time_sig_valid = model(input_x_valid)
        output_valid = output_valid.squeeze()
        input_x_valid = input_x_valid.squeeze(0)

        #
        # get the desired result
        target_valid = Label_validation[idx, :]
        target_valid = torch.FloatTensor(target_valid)
        target_valid = target_valid.to(device)

        if( (analysis_bank == True) & (synthesis_bank==False)):
            loss = loss_fct(output_valid, target_valid)
            valid_loss.append(loss.item())
        if( (analysis_bank == False) & (synthesis_bank==True)):
            loss_time_signal = loss_fct(time_sig_valid, float(model.nb_freq_pts) * input_x_valid)
            valid_loss.append(loss_time_signal.item())

    return train_loss, valid_loss


# This is the main Epoch loop
# ---------------------------
epoch_train_loss, epoch_valid_loss = [], []
for epoch in range(nb_batches):  # loop over the dataset multiple times
    # get training data
    # each epoch , use a batch : 2D matrix
    sig = dataset[epoch]
    sig_validation = dataset_validation[epoch]

    print(f'size of sig = {sig.shape}')
    
    # get the expected output from the library FFT
    sig_fft = np.fft.rfft(sig, axis=-1)
    # 2D desired output matrix:  each row: N real, N imag
    Label = np.hstack([sig_fft.real, sig_fft.imag])

    # validation
    sig_validation_fft = np.fft.rfft(sig_validation, axis=-1)
    # 2D desired output matrix:  each row: N/2 real, N/2 imag
    Label_validation = np.hstack([sig_validation_fft.real, sig_validation_fft.imag])
#
    # First 200 epoch to train analysis bank
    if( epoch < 200):
        loss_function = nn.MSELoss()
        optimizer = optim.SGD(c_dft.parameters(), lr=0.01, momentum=0.9)  # for case of no initialization
        train_loss, valid_loss = train_banks(model=c_dft, epoch_ind=epoch, optimizer=optimizer, scheduler=None, loss_fct=loss_function, sig=sig, Label=Label, batch_size=batch_size, sig_validation=sig_validation, Label_validation= Label_validation, N=NFFT, analysis_bank=True, synthesis_bank=False)
    #
    # Next epochs to train synthesis bank
    else:
        loss_function = nn.SmoothL1Loss()
        if (epoch > 350):
            optimizer = optim.SGD(c_dft.parameters(), lr=0.005, momentum=0.75)  # 0.01 and 0.75 is the best
        if (epoch > 450):
            optimizer = optim.SGD(c_dft.parameters(), lr=0.001, momentum=0.75)  # 0.01 and 0.75 is the best
        if (epoch > 550):
            optimizer = optim.SGD(c_dft.parameters(), lr=0.0002, momentum=0.75)  # 0.01 and 0.75 is the best
        if (epoch <= 350):
            optimizer = optim.SGD(c_dft.parameters(), lr=0.01, momentum=0.75)  # 0.01 and 0.75 is the best
        
        train_loss, valid_loss = train_banks(model=c_dft, epoch_ind=epoch, optimizer=optimizer, scheduler=None, loss_fct=loss_function, sig=sig, Label=Label, batch_size=batch_size, sig_validation=sig_validation, Label_validation= Label_validation, N=NFFT, analysis_bank=False, synthesis_bank=True)

    # print sum of weights
    print(torch.sum(c_dft.conv1r.weight.data))
    epoch_train_loss.append(np.mean(train_loss))
    epoch_valid_loss.append(np.mean(valid_loss))
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))


# plot loss curve

epoch_nb = np.arange(0,nb_batches,1)
epoch_train_loss = np.array(epoch_train_loss)
epoch_valid_loss = np.array(epoch_valid_loss)
print(f'epoch nb {epoch_nb.shape}, {epoch_train_loss.shape}')
plt.figure(figsize=(6, 5));
plt.plot(epoch_nb, epoch_train_loss )
plt.xlabel('Epoch Nb');
plt.ylabel('Training Loss (Lin Scale)');
plt.xlabel('Epoch Nb');
plt.savefig('Training_loss_lin.png')


plt.figure(figsize=(6, 5));
plt.plot(epoch_nb, np.log(epoch_train_loss + 1.0e-15) )
plt.xlabel('Epoch Nb');
plt.ylabel('Training Loss (Log Scale)');
plt.xlabel('Epoch Nb');
plt.savefig('Training_loss_log.png')

print(f'size of kernels = {real_kernels.shape}')
print(f'size of kernels = {imag_kernels.shape}')


# Plot the finals kernels
plot_final_kernels(model=c_dft, NFFT=NFFT, hop_length=NFFT//2, sr=sample_rate, filename_kernels='Final_Kernels', ini_r_kernels = real_kernels.squeeze(), ini_i_kernels=imag_kernels.squeeze(), ini_inv_r_kernels = real_inv_kernels.squeeze(), ini_inv_i_kernels=imag_inv_kernels.squeeze())

# Run eval
evaluate_DFT_IDFT(model=c_dft, NFFT=NFFT, hop_length=NFFT//2,  eval_filename='Final_eval_dft_idft')

# Show all figures
plt.show()

