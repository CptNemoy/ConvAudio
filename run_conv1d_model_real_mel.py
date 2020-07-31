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
from speech_melgram import create_mel_filters
from eval_model import evaluate_melgram
from eval_model import plot_initial_kernels
from eval_model import plot_final_kernels


device = 'cpu'

# create a dataset of several batches of complex vectors of size N

N = 1024
NFFT = N
hop_length = int(N/2)
nfilt = 64
sample_rate = 22000
nb_batches = 600         # nb of epochs
batch_size = 400         # nb of samples used in each epoch
dataset = ComplexNumbersDataset(nb_batches, batch_size, N, complex=False)
dataset_validation = ComplexNumbersDataset(nb_batches, batch_size, N, complex=False)

print(f' dataset of 0 = {dataset[0]}')

# initialize model
c_dft = ConvDFT(filter_length=N, hop_length=N/2, win_length=None,window='hann', n_mels=nfilt)
print(c_dft)


cutoff = int((N / 2) + 1)

# Get the Mel filters
fbank = create_mel_filters(sample_rate=sample_rate, n_fft=NFFT, nfilt=nfilt)


# evaluate the model before we start training
evaluate_melgram(model=c_dft, NFFT=NFFT, hop_length=hop_length, fbank=fbank, eval_filename='initial_model_melgram.png')

# Plot the initial kernels
plot_initial_kernels(model=c_dft, NFFT=NFFT, hop_length=hop_length, sr=sample_rate, fbank=fbank, filename_kernels='Initial_Kernels.png', filename_mels='Initial_Mels.png')





#exit()


loss_function = nn.SmoothL1Loss()
loss_function_mse = nn.MSELoss()


c_dft.conv1r.weight.requires_grad = True
c_dft.conv1r.bias.requires_grad = True
c_dft.conv1i.weight.requires_grad = True
c_dft.conv1i.bias.requires_grad = True

c_dft.conv1d_mels.weight.requires_grad = True
c_dft.conv1d_mels.bias.requires_grad = True



#optimizer = optim.SGD(c_dft.parameters(), lr=0.0001, momentum=0.9)  # for case of  initialization/no initialization
#optimizer = optim.SGD(c_dft.parameters(), lr=0.001, momentum=0.8) #  no initialization
# adam optimizer lr=0.001 very slowly converging for non-initialized
#optimizer = optim.Adam(c_dft.parameters(), lr=0.0001, betas=(0.8, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  # no initialization

#optimizer = optim.Adam(c_dft.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) # for initialized kernels

optimizer = optim.SGD(c_dft.parameters(), lr=0.1, momentum=0.8)  # for case of init cos/sin kernels
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.25)

epoch_train_loss, epoch_valid_loss = [], []
for epoch in range(nb_batches):  # loop over the dataset multiple times
    c_dft.train()
    scheduler.step()

    # get training data
    # each epoch , use a batch : 2D matrix make
    sig = dataset[epoch]
    sig_validation = dataset_validation[epoch]

    #print(f'size of sig = {sig.shape}')
    # input matrix: each row: N real pts
    X = sig.real
    #print(f'shape of X = {X.shape}')
    
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


    # the FFT of a real input is redundant, keep only (N/2+1) pts from each real, imag
    train_loss, valid_loss = [], []

    
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
        #print(f'shape of input_x = {input_x.shape}')

        # run forward
        # Note : for now, we assume the input consists of 1 frame of data of length N
        output, mel_out, power_frame = c_dft(input_x)
        output = output.squeeze()
        
        mel_out = mel_out.squeeze() / float(NFFT)
        #print(f'shape of output = {output.shape}. mel_out {mel_out.shape}')

        # get the desired result
        target = Label[idx, :]              # FFT output
        target_mel = MelgramBatch[idx, :]   # Mel output
                
        #target = torch.from_numpy(target).float().to(device)
        target = torch.FloatTensor(target)
        target = target.to(device)

        target_mel = torch.FloatTensor(target_mel)
        target_mel = target_mel.to(device)

        #print(f'shape of output_mel = {mel_out.shape}, shape of target = {target_mel.shape}')

        #print(f'shape of output = {output.shape}, shape of target = {target.shape}')

        # get error
        loss = loss_function(output, target)
        loss_mel =  loss_function(mel_out, target_mel)
        
        total_loss = loss + loss_mel
        
        #print(f'loss_mel = {loss_mel}')
        train_loss.append(total_loss.item())
        
        # backward prop
        total_loss.backward()
        optimizer.step()

    ## evaluation part
    c_dft.eval()
    for idx in range(batch_size):
        # get one row of input
        input_x_valid = sig_validation[idx, :]
        #
        #input_x = torch.from_numpy(input_x).float().to(device)
        # input_x has to be in [1, L]    shape
        input_x_valid = torch.FloatTensor(input_x_valid)
        input_x_valid = input_x_valid.unsqueeze(0)
        input_x_valid = input_x_valid.to(device)
        #
        #print(f' input valid shape ={input_x_valid.shape}')
        output_valid, mel_out_valid, power_frame = c_dft(input_x_valid)
        output_valid = output_valid.squeeze()
        mel_out_valid = mel_out_valid.squeeze() / float(NFFT)
        
        #print(f' output valid shape ={output_valid.shape}')

        #
        # get the desired result
        target_valid = Label_validation[idx, :]
        target_valid_mel = MelgramBatchValidation[idx, :]
        #
        #target = torch.from_numpy(target).float().to(device)
        target_valid = torch.FloatTensor(target_valid)
        target_valid = target_valid.to(device)
        
        target_valid_mel =torch.FloatTensor(target_valid_mel)
        target_valid_mel = target_valid_mel.to(device)

        #
        loss = loss_function(mel_out_valid, target_valid_mel)
        valid_loss.append(loss.item())

    
    # print sum of weights
    print(torch.sum(c_dft.conv1r.weight.data))
    #print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss) )
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
plt.ylabel('Training Loss');
plt.savefig('Training_loss.png')
plt.show()
#summary(c_dft, input_data=(34,N))
#summary(c_dft)


# Plot the final kernels
plot_final_kernels(model=c_dft, NFFT=NFFT, hop_length=hop_length, sr=sample_rate, fbank=fbank, filename_kernels='Final_Kernels.png', filename_mels='Final_Mels.png')


evaluate_melgram(model=c_dft, NFFT=NFFT, hop_length=hop_length, fbank=fbank, eval_filename='model_melgram.png')




exit()
