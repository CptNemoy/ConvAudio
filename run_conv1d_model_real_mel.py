import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from create_datasets import ComplexNumbersDataset
from Torch_DFT_Conv1d_real_Mel import ConvDFT
from torchsummary import summary
import librosa
from speech_melgram import get_filterbanks
from eval_model import evaluate_melgram
from eval_model import plot_initial_kernels
from eval_model import plot_final_kernels


device = 'cpu'

# System Parameters
# =================
N = 1024
NFFT = N
hop_length = int(N/2)
nfilt = 64
sample_rate = 22050
nb_batches = 600         # nb of epochs
batch_size = 400         # nb of samples used in each epoch
init_dft_kernels=False
init_mel_kernels=False
# ----------------------

# create a dataset of several batches of complex vectors of size N
dataset = ComplexNumbersDataset(nb_batches, batch_size, N, complex=False)
dataset_validation = ComplexNumbersDataset(nb_batches, batch_size, N, complex=False)

# initialize model
c_dft = ConvDFT(filter_length=N, hop_length=hop_length, win_length=None,window='hann', n_mels=nfilt, sr=sample_rate, init_dft_kernels=init_dft_kernels, init_mel_kernels=init_mel_kernels)
print(c_dft)

# Get the Mel filters
fbank = get_filterbanks(samplerate=sample_rate, nfft=NFFT, nfilt=nfilt)

# evaluate the model before we start training
evaluate_melgram(model=c_dft, NFFT=NFFT, hop_length=hop_length, fbank=fbank, eval_filename='initial_model_melgram')

# Plot the initial kernels
plot_initial_kernels(model=c_dft, NFFT=NFFT, hop_length=hop_length, sr=sample_rate, fbank=fbank, n_mels=nfilt, filename_kernels='Initial_Kernels', filename_mels='Initial_Mels.png')


# get the initial mels for later comparison
initial_mel_kernels = c_dft.conv1d_mels.weight.data
initial_mel_kernels = initial_mel_kernels.squeeze()
i_m_kernels = np.array(initial_mel_kernels)


# Select loss function and optimizers
loss_function_mse = nn.MSELoss()
loss_function = nn.SmoothL1Loss()

if( (init_dft_kernels==True) & (init_mel_kernels==True)):
    optimizer = optim.SGD(c_dft.parameters(), lr=0.001, momentum=0.8)  # for case of init cos/sin kernels
else:
    optimizer = optim.Adagrad(c_dft.parameters(), lr=0.1, lr_decay=5e-06, weight_decay=1e-06, initial_accumulator_value=0)  #0.00029755302850389856


# Training Function , called for each Epoch
# Return value : training and validation loss for the batch
#
def train_banks(model, epoch_ind, optimizer, scheduler, loss_fct, sig, Label, batch_size, sig_validation, Label_validation, N, dft_bank=False, mel_bank=False):

    if( (dft_bank == False) & (mel_bank==False)):
        print(f'Nothing to train')
        return 0, 0
    if( (dft_bank == True) & (mel_bank==False)):
        model.conv1r.weight.requires_grad = True
        model.conv1r.bias.requires_grad = True
        model.conv1i.weight.requires_grad = True
        model.conv1i.bias.requires_grad = True
        #
        model.conv1d_mels.weight.requires_grad = False
        model.conv1d_mels.bias.requires_grad = False
        #
    if( (dft_bank == False) & (mel_bank==True)):
        model.conv1r.weight.requires_grad = False
        model.conv1r.bias.requires_grad = False
        model.conv1i.weight.requires_grad = False
        model.conv1i.bias.requires_grad = False
        #
        model.conv1d_mels.weight.requires_grad = True
        model.conv1d_mels.bias.requires_grad = True
        #
    if( (dft_bank == True) & (mel_bank==True)):
        model.conv1r.weight.requires_grad = True
        model.conv1r.bias.requires_grad = True
        model.conv1i.weight.requires_grad = True
        model.conv1i.bias.requires_grad = True
        #
        model.conv1d_mels.weight.requires_grad = True
        model.conv1d_mels.bias.requires_grad = True
        #

    train_loss, valid_loss = [], []
    model.train()
    
    for idx in range(batch_size):
        # get one row of input
        input_x = sig[idx, :]
        
        # zero gradients
        optimizer.zero_grad()

        # input_x has to be in [1, L]    shape
        input_x = torch.FloatTensor(input_x)
        input_x = input_x.unsqueeze(0)
        input_x = input_x.to(device)

        # run forward
        # Note : for now, we assume the input consists of 1 frame of data of length N
        output, mel_out, power_frame = model(input_x)
        output = output.squeeze()
        
        mel_out = mel_out.squeeze() / float(NFFT)

        # get the desired result
        target = Label[idx, :]              # FFT output
        target = torch.FloatTensor(target)
        target = target.to(device)

        target_mel = MelgramBatch[idx, :]   # Mel output
        target_mel = torch.FloatTensor(target_mel)
        target_mel = target_mel.to(device)

        # get error
        loss = loss_fct(output, target)
        loss_mel =  loss_fct(mel_out, target_mel)

        total_loss = loss + loss_mel
 
        if( (dft_bank == True) & (mel_bank==False)):
            train_loss.append(loss.item())
            # backward prop
            loss.backward()
        if( (dft_bank == False) & (mel_bank==True)):
            train_loss.append(loss_mel.item())
            # backward prop
            loss_mel.backward()
        if( (dft_bank == True) & (mel_bank==True)):
            train_loss.append(total_loss.item())
            # backward prop
            total_loss.backward()
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
        output_valid, mel_out_valid, power_frame = model(input_x_valid)
        output_valid = output_valid.squeeze()
        mel_out_valid = mel_out_valid.squeeze() / float(NFFT)
        #
        # get the desired result
        target_valid = Label_validation[idx, :]
        target_valid_mel = MelgramBatchValidation[idx, :]
        # tensor
        target_valid = torch.FloatTensor(target_valid)
        target_valid = target_valid.to(device)
        # tensor
        target_valid_mel =torch.FloatTensor(target_valid_mel)
        target_valid_mel = target_valid_mel.to(device)
        # Compute Loss
        loss = loss_fct(mel_out_valid, target_valid_mel)
        valid_loss.append(loss.item())

    # return the arrays of loss for the entire batch
    return train_loss, valid_loss



# ===============
# Main Epoch Loop
# ===============
epoch_train_loss, epoch_valid_loss = [], []
for epoch in range(nb_batches):  # loop over the dataset multiple times

    # get training data
    # each epoch , use a batch : 2D matrix make
    sig = dataset[epoch]
    sig_validation = dataset_validation[epoch]

    # input matrix: each row: N real pts
    X = sig.real
    
    # get the expected output from the library FFT
    # sig is a 2D array : [frames, N-pt vector]
    #
    sig_fft = np.fft.rfft(sig, axis=-1)
    # 2D desired FFT matrix:  each row: N/2 real, N/2 imag
    Label = np.hstack([sig_fft.real, sig_fft.imag])
    print(f'shape of Label = {Label.shape}')
    
    # construct Melgram
    #mag_frames = np.absolute(Label)                  # Magnitude of the FFT
    pow_frames =  ( ( (sig_fft.real) ** 2) + ((sig_fft.imag) ** 2) )       # Power Spectrum
    print(f'shape of pow frames = {pow_frames.shape}')
    print(f'shape of fbank.T = {fbank.shape}.   .T = {fbank.T.shape}')
    
    
    MelgramBatch = np.dot(pow_frames, fbank.T) / float(NFFT)
    print(f'size of MelgramBatch = {MelgramBatch.shape}')
    
    # validation
    sig_validation_fft = np.fft.rfft(sig_validation, axis=-1)
    # 2D desired output matrix:  each row: N/2 real, N/2 imag
    Label_validation = np.hstack([sig_validation_fft.real, sig_validation_fft.imag])
    print(f' size of label validation = {Label_validation.shape}')
    pow_frames_validation = ( ((sig_validation_fft.real) ** 2) + ((sig_validation_fft.imag) ** 2)    )   # Power Spectrum
    print(f'shape of pow frames = {pow_frames.shape}')
    print(f'shape of fbank.T = {fbank.shape}.   .T = {fbank.T.shape}')
    MelgramBatchValidation = np.dot(pow_frames_validation, fbank.T) / float(NFFT)

    # call training function on all frames in batch
    train_loss, valid_loss = train_banks(model=c_dft, epoch_ind=epoch, optimizer=optimizer, scheduler=None, loss_fct=loss_function, sig=sig, Label=Label, batch_size=batch_size, sig_validation=sig_validation, Label_validation= Label_validation, N=NFFT, dft_bank=True, mel_bank=True)

    # print sum of weights
    print(torch.sum(c_dft.conv1r.weight.data))

    epoch_train_loss.append(np.mean(train_loss))
    epoch_valid_loss.append(np.mean(valid_loss))
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))

# plot loss curve
epoch_nb = np.arange(0,nb_batches,1)
epoch_train_loss = np.array(epoch_train_loss)
print(f'epoch nb {epoch_nb.shape}, {epoch_train_loss.shape}')
plt.figure(figsize=(8, 4));
plt.plot(epoch_nb, epoch_train_loss)
plt.xlabel('Epoch');
plt.ylabel('Training Loss (Lin)');
plt.savefig('Training_loss_lin.png')

plt.figure(figsize=(8, 4));
plt.plot(epoch_nb, np.log(epoch_train_loss+1.0e-10))
plt.xlabel('Epoch');
plt.ylabel('Training Loss (Log)');
plt.savefig('Training_loss_log.png')

# Plot the final kernels
plot_final_kernels(model=c_dft, NFFT=NFFT, hop_length=hop_length, sr=sample_rate, fbank=fbank, n_mels=nfilt, filename_kernels='Final_Kernels', filename_mels='Final_Mels', initial_mels = i_m_kernels)


evaluate_melgram(model=c_dft, NFFT=NFFT, hop_length=hop_length, fbank=fbank, eval_filename='final_model_melgram')

# show all figures
plt.show()

# bye
exit()
