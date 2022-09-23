import os
import sys
import argparse
import numpy as np
import soundfile as sf
import librosa
import resampy
import datetime
import _pickle as pickle
from Torch_DFT_Conv1d_real_Chroma import ConvDFT
from torchsummary import summary
import librosa
import torch
import matplotlib.pyplot as plt
from speech_chroma import create_chroma_filters
from eval_model_chroma import evaluate_chroma

device = 'cpu'


def initialize_model(NFFT, sr, n_filt, hop_length):
    # ideally, initialization should be called once only, but right now, it's not working.
    init_dft_kernels=True
    init_chroma_kernels=True

    # initialize model
    c_dft = ConvDFT(filter_length=NFFT, hop_length=hop_length, win_length=None, window='hann', n_chroma=n_filt, sr=sr, init_dft_kernels=init_dft_kernels, init_chroma_kernels=init_chroma_kernels)


    return c_dft



def  evaluate_melgram(model, NFFT, n_filt, hop_length, audio_segment, sr):
    # This method initializes the model (loads the kernels for dft/chroma) and then calls it for a given audio signal
    # the chromagram is returned as a 2D matrix of values
    #
    #
    init_dft_kernels=True
    init_chroma_kernels=True

    # initialize model
    c_dft = ConvDFT(filter_length=NFFT, hop_length=hop_length, win_length=None, window='hann', n_chroma=n_filt, sr=sr, init_dft_kernels=init_dft_kernels, init_chroma_kernels=init_chroma_kernels)

    c_dft.eval()
    
    # call librosa chroma directly on the original signal
    chroma_signal = librosa.feature.chroma_stft(y=audio_segment, sr=sr, S=None, n_fft=NFFT, hop_length=hop_length, tuning=0.0, n_chroma=n_filt, norm=4.0, window='boxcar' )

    m_audio = torch.FloatTensor(audio_segment)
    m_audio = m_audio.unsqueeze(0)
    m_audio = m_audio.to(device)

    print(f'size of audio = {audio.shape}')
    #
    #
    # Call model
    output_eval, chroma_out_eval, power_frames_eval = model(m_audio)

    # back to numpy array and normalize
    chroma_out_eval = chroma_out_eval.T.squeeze()
    c_frames = chroma_out_eval.cpu().data.numpy()
    c_frames = librosa.util.normalize(c_frames, norm=4.0, axis=0)

    
    # plot and compare the librosa w/ the model
    #
    #plt.figure(figsize=(6, 4));
    #plt.plot(audio_segment);
    
    # Plot the 3D Chromagram: Model vs. Librosa
    plt.figure(figsize=(9, 7));
    plt.subplot(211);
    plt.imshow((1.0+c_frames), aspect='auto', origin='lower')
    plt.title('Model - Chromagram of Audio Seg');
    plt.ylabel('Chroma Filter Nb');
    plt.xlabel('Time (Frames)');
    plt.colorbar();
    plt.subplot(212);
    plt.imshow((1.0+chroma_signal), aspect='auto', origin='lower')
    plt.title('Librosa Chromagram of Audio Seg');
    plt.ylabel('Chroma Filter Nb');
    plt.xlabel('Time (Frames)');
    plt.colorbar();
    plt.show()
    
    return c_frames




# convert number of samples to HH:MM:SS.ss string
def duration(nsamp, sr):
  return str(datetime.timedelta(seconds = nsamp / sr))

# scan files from source directory and build list with info for each file
def scan_wfiles(src, gender):
  list = []
  for fn in os.listdir(src):
    path = os.path.join(src, fn)
    if os.path.isdir(path):
      list += scan_wfiles(path, gender)
    elif fn.endswith('wav'):
      with sf.SoundFile(path, "r") as wfin:
        sr = wfin.samplerate
        frames = wfin.frames
        channels = wfin.channels
      f = {"path": path, "gender": gender, "sr": sr, "frames": frames, "channels": channels}
      list.append(f)
  return list

if __name__ == '__main__':
  ap = argparse.ArgumentParser(
    description='Gender Detection Preprocess')
  ap.add_argument('-i', '--InDir', action="store", dest='din',
                  required=True, help='Input source directory')
  ap.add_argument('-o', '--OutDir', action="store", dest='dout',
                  required=True, help='Output gram directory')
  ap.add_argument('-l', '--Label', action="store", dest='label',
                  required=True, help='Label for every file (M, F, N)')
                  
  args = ap.parse_args()

  label = args.label

  
  NFFT = 1024
  samplerate = 16000
  hop_length = int(NFFT / 2)
  block_size = 64  # 1024/16k frame with 50% overlap (hop_length) = 2.048sec
  block_overlap = int(block_size / 2)
  

  my_model = initialize_model(NFFT=NFFT, sr=samplerate, n_filt=64, hop_length=hop_length)

  
  
  l = scan_wfiles(args.din, label)
  # get some stats
  cnt = [0,0,0]
  dur = [0,0,0]
  sr = 0
  
  print(f'l = {l}')
  
  for iter in l:
    type = "MFN".find(iter['gender'])
    cnt[type] += 1
    dur[type] += iter['frames']
    if sr==0:
      sr = iter['sr']
    else:
      if sr != iter['sr']:
        print(f"sr change from {sr} to {iter['sr']}")
        sr = iter['sr']
  print(f'Number of files: Male={cnt[0]}, Female={cnt[1]}, Neither={cnt[2]}')
  print(f'Duration: Male = {duration(dur[0],sr)}, Female={duration(dur[1],sr)}, Neither={duration(dur[2],sr)}')
  dout = args.dout
  if not os.path.exists(dout):
    print(f'Creating output directory {dout}')
    os.mkdir(dout)
  cnt = 0
  for iter in l:
    #audio, sr = librosa.load('got_s2e9_cake.wav')
    audio,sr = librosa.load(iter['path'])
    audio = audio * 20.0        # must amplify audio to make model work
    if sr != samplerate:  # resample to target rate
      audio = resampy.resample(audio, sr, samplerate)
      sr = samplerate
      iter['sr'] = samplerate
      
    # melgram
    librosa_mel = librosa.feature.melspectrogram(y=audio, sr=sr, S=None, n_fft=NFFT, hop_length=int(hop_length), power=4.0, n_mels=64, window='boxcar' )
    # Chromagram
    librosa_chroma = librosa.feature.chroma_stft(y=audio, sr=sr, S=None, n_fft=NFFT, hop_length=int(hop_length), tuning=0.0, n_chroma=64, norm=4.0, window='boxcar' )
    
    model_chroma = evaluate_melgram(model=my_model, NFFT=NFFT, n_filt=64, hop_length=int(hop_length), audio_segment=audio, sr=sr)
    
    print(f'size of librosa_chroma = {librosa_chroma.shape}')
    print(f'size of model_chroma = {model_chroma.shape}')

    #print(f'***Shape={librosa_mel.shape}')
    for idx in range(0,librosa_mel.shape[1]-block_size,block_overlap):
      meta = iter
      meta['idx'] = idx
      #meta['data'] = librosa_chroma[:,idx:idx+block_size]
      meta['data'] = model_chroma[:,idx:idx+block_size]
      #print(f"size={meta['data'].shape}")
      fn = f'mg{cnt}.p'
      opath = os.path.join(dout, fn)
      pickle.dump(meta, open(opath, "wb"))
      cnt += 1
  print(f'Created {cnt} files')
