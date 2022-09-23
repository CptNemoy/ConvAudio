Place all .py file in the main folder 

edit the parameters in file :       run_conv1d_model.py

NFFT = 1024
nb_batches = 800
batch_size = 200
sample_rate = 16000
init_dft_kernels=False       # True if we wish to initialize the kernels
init_idft_kernels=False



then start : 

python run_conv1d_model.py


it will run for all the epochs, and then will plot a bunch of figures as well as save them to disk.


