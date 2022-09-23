import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from create_datasets import ComplexNumbersDataset
from Torch_DFT_Conv1d_real_Chroma import ConvDFT
from torchsummary import summary
import librosa
from speech_chroma import create_chroma_filters
from eval_model_chroma import evaluate_chroma
from eval_model_chroma import plot_initial_kernels
from eval_model_chroma import plot_final_kernels


device = 'cpu'


# User changeable parameters
# ==========================
N = 1024
NFFT = N
hop_length = int(N/2)
n_chroma = 24
sample_rate = 22000
nb_batches = 200         # nb of epochs
batch_size = 400         # nb of samples used in each epoch
init_dft_kernels=True
init_chroma_kernels=True
# ---------------------------


# create a dataset of several batches of complex vectors of size N
dataset = ComplexNumbersDataset(nb_batches, batch_size, N, complex=False)
dataset_validation = ComplexNumbersDataset(nb_batches, batch_size, N, complex=False)

# initialize model
c_dft = ConvDFT(filter_length=N, hop_length=N/2, win_length=None, window='hann', n_chroma=n_chroma, sr=sample_rate, init_dft_kernels=init_dft_kernels, init_chroma_kernels=init_chroma_kernels)
print(c_dft)

# Get the Chroma filters
fbank = create_chroma_filters(sample_rate=sample_rate, n_fft=NFFT, n_chroma=n_chroma)

# evaluate the model before we start training
evaluate_chroma(model=c_dft, NFFT=NFFT, hop_length=hop_length, fbank=fbank, n_chroma=n_chroma, eval_filename='initial_model_chroma_w_init', librosa_filename='initial_librosa_chroma_w_init')

# Plot the initial kernels
plot_initial_kernels(model=c_dft, NFFT=NFFT, hop_length=hop_length, sr=sample_rate, fbank=fbank, filename_kernels='Initial_Kernels_w_init', filename_chroma='Initial_Chroma_w_init.png', n_chroma=n_chroma)

# get the initial mels for later comparison
initial_chroma_kernels = c_dft.conv1d_chroma.weight.data
initial_chroma_kernels = initial_chroma_kernels.squeeze()
i_c_kernels = np.array(initial_chroma_kernels)


loss_function_mse = nn.MSELoss()
loss_function = nn.SmoothL1Loss()

if (init_dft_kernels==True) & (init_chroma_kernels==True) :
    optimizer = optim.SGD(c_dft.parameters(), lr=0.001, momentum=0.8)  # for case of init cos/sin , chroma kernels
else:
    optimizer = optim.Adagrad(c_dft.parameters(), lr=0.5, lr_decay=1e-06, weight_decay=1e-06, initial_accumulator_value=0)  # no initialization    FINAL                Epoch: 699 Training Loss:  0.000663621528656222.   chromagram looks good.


# function called for each batch
#
def train_banks(model, epoch_ind, sig, Label, ChromaBatch, batch_size, loss_fct, sig_validation, Label_validation, ChromaBatchValidation, N, dft_bank=False, chroma_bank=False):

    if( (dft_bank == False) & (chroma_bank==False)):
        print(f'Nothing to train')
        return 0, 0
    if( (dft_bank == True) & (chroma_bank==False)):
        model.conv1r.weight.requires_grad = True
        model.conv1r.bias.requires_grad = True
        model.conv1i.weight.requires_grad = True
        model.conv1i.bias.requires_grad = True
        #
        model.conv1d_chroma.weight.requires_grad = False
        model.conv1d_chroma.bias.requires_grad = False
        #
    if( (dft_bank == False) & (chroma_bank==True)):
        model.conv1r.weight.requires_grad = False
        model.conv1r.bias.requires_grad = False
        model.conv1i.weight.requires_grad = False
        model.conv1i.bias.requires_grad = False
        #
        model.conv1d_chroma.weight.requires_grad = True
        model.conv1d_chroma.bias.requires_grad = True
        #
    if( (dft_bank == True) & (chroma_bank==True)):
        model.conv1r.weight.requires_grad = True
        model.conv1r.bias.requires_grad = True
        model.conv1i.weight.requires_grad = True
        model.conv1i.bias.requires_grad = True
        #
        model.conv1d_chroma.weight.requires_grad = True
        model.conv1d_chroma.bias.requires_grad = True
        #
        

    train_loss, valid_loss = [], []
    model.train()
        
    for idx in range(batch_size):

        # zero gradients
        optimizer.zero_grad()

        # get one row of input
        input_x = X[idx, :]
        
        #input_x = torch.from_numpy(input_x).float().to(device)
        # input_x has to be in [1, L]    shape
        input_x = torch.FloatTensor(input_x)
        input_x = input_x.unsqueeze(0)
        input_x = input_x.to(device)

        # run forward
        # Note : for now, we assume the input consists of 1 frame of data of length N
        output, chroma_out, power_frame = model(input_x)
        output = output.squeeze()
        
        chroma_out = chroma_out.squeeze() / float(NFFT)

        # get the desired result
        target = Label[idx, :]              # FFT output
        target = torch.FloatTensor(target)
        target = target.to(device)

        target_chroma = ChromaBatch[idx, :]   # Mel output
        target_chroma = torch.FloatTensor(target_chroma)
        target_chroma = target_chroma.to(device)

        # get error
        loss = loss_fct(output, target)
        loss_chroma =  loss_fct(chroma_out, target_chroma)
        
        total_loss = loss + loss_chroma

        if( (dft_bank == True) & (chroma_bank==False)):
            train_loss.append(loss.item())
            # backward prop
            loss.backward()
        if( (dft_bank == False) & (chroma_bank==True)):
            train_loss.append(loss_chroma.item())
            # backward prop
            loss_chroma.backward()
        if( (dft_bank == True) & (chroma_bank==True)):
            train_loss.append(total_loss.item())
            # backward prop
            total_loss.backward()

        train_loss.append(total_loss.item())
        
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
        output_valid, mel_out_valid, power_frame = c_dft(input_x_valid)
        output_valid = output_valid.squeeze()
        mel_out_valid = mel_out_valid.squeeze() / float(NFFT)
        
        #
        # get the desired result
        target_valid = Label_validation[idx, :]
        target_valid_chroma = ChromaBatchValidation[idx, :]

        #
        target_valid = torch.FloatTensor(target_valid)
        target_valid = target_valid.to(device)
        
        target_valid_chroma =torch.FloatTensor(target_valid_chroma)
        target_valid_chroma = target_valid_chroma.to(device)

        #
        loss = loss_fct(mel_out_valid, target_valid_chroma)
        valid_loss.append(loss.item())


    return train_loss, valid_loss




# -------------------------
# Main Epoch Loop
# -------------------------
epoch_train_loss, epoch_valid_loss = [], []
for epoch in range(nb_batches):  # loop over the dataset multiple times
    c_dft.train()

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
    pow_frames =  ( ( (sig_fft.real) ** 2) + ((sig_fft.imag) ** 2) )       # Power Spectrum
    ChromaBatch = np.dot(pow_frames, fbank.T) / float(NFFT)
    
    # validation
    sig_validation_fft = np.fft.rfft(sig_validation, axis=-1)
    # 2D desired output matrix:  each row: N/2 real, N/2 imag
    Label_validation = np.hstack([sig_validation_fft.real, sig_validation_fft.imag])
    pow_frames_validation = ( ((sig_validation_fft.real) ** 2) + ((sig_validation_fft.imag) ** 2)    )   # Power Spectrum

    ChromaBatchValidation = np.dot(pow_frames_validation, fbank.T) / float(NFFT)

    # call training function
    train_loss, valid_loss = train_banks(model=c_dft, epoch_ind=epoch, sig=sig, Label=Label, loss_fct=loss_function, ChromaBatch=ChromaBatch, batch_size=batch_size, sig_validation=sig_validation, Label_validation=Label_validation, ChromaBatchValidation=ChromaBatchValidation, N=N, dft_bank=True, chroma_bank=True)

    
    # print sum of weights
    print(torch.sum(c_dft.conv1r.weight.data))
    epoch_train_loss.append(np.mean(train_loss))
    epoch_valid_loss.append(np.mean(valid_loss))
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))

# plot loss curve

epoch_nb = np.arange(0,nb_batches,1)
epoch_train_loss = np.array(epoch_train_loss)
print(f'epoch nb {epoch_nb.shape}, {epoch_train_loss.shape}')
plt.figure(figsize=(10, 4));
plt.plot(epoch_nb, epoch_train_loss)
plt.xlabel('Epoch');
plt.ylabel('Training Loss (Lin)');
plt.savefig('Training_loss_lin.png')

plt.figure(figsize=(10, 4));
plt.plot(epoch_nb, np.log(epoch_train_loss+1.0e-10))
plt.xlabel('Epoch');
plt.ylabel('Training Loss (Log)');
plt.savefig('Training_loss_log.png')


# Plot the final kernels
plot_final_kernels(model=c_dft, NFFT=NFFT, hop_length=hop_length, sr=sample_rate, fbank=fbank, n_chroma=n_chroma,filename_kernels='Final_Kernels_w_init', filename_chroma='Final_Chroma_w_init.png', filename_chroma_filters='Final_Filter_coefficients', initial_c_kernels= i_c_kernels)

evaluate_chroma(model=c_dft, NFFT=NFFT, hop_length=hop_length, fbank=fbank, n_chroma=n_chroma, eval_filename='final_model_chroma_w_init.png', librosa_filename='final_librosa_chroma_w_init.png')

# Show all figures
plt.show()

# bye
exit()
