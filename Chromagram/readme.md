Place all .py file in the main folder 

edit the parameters in file :       run_conv1d_model.py

N = 1024
NFFT = N
hop_length = int(N/2)
n_chroma = 24
sample_rate = 22000
nb_batches = 800         # nb of epochs
batch_size = 400         # nb of samples used in each epoch
init_dft_kernels=False
init_chroma_kernels=False


then start : 

python run_conv1d_model_real_chroma.py


it will run for all the epochs, and then will plot a bunch of figures as well as save them to disk.



