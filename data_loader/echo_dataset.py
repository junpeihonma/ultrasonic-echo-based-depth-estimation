import os.path
import time
import librosa
import h5py
import random
import math
import numpy as np
import glob
import torch
import csv
import pickle
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torch.utils.data as data
from scipy.signal import chirp, spectrogram
from scipy import signal
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.transforms as T
from torchvision.io import read_image
import cv2

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return samples

# STFT
def generate_spectrogram(echo1, echo2, echo3, echo4, winl=32):
    stft_echo1 = librosa.stft(echo1, n_fft=512, win_length=winl)
    stft_echo2 = librosa.stft(echo2, n_fft=512, win_length=winl)
    stft_echo3 = librosa.stft(echo3, n_fft=512, win_length=winl)
    stft_echo4 = librosa.stft(echo4, n_fft=512, win_length=winl)

    return np.concatenate((np.expand_dims(np.abs(stft_echo1), axis=0), np.expand_dims(np.abs(stft_echo2), axis=0),  np.expand_dims(np.abs(stft_echo3), axis=0),  np.expand_dims(np.abs(stft_echo4), axis=0)), axis=0)

def parse_all_data(root_path,mode,dataset):
    """ Read each echo's path """
    with open(root_path, encoding='utf8', newline='') as f:
        csvreader = csv.reader(f)
        content = [row for row in csvreader]

    # Splitting the dataset
    if dataset == 'TUS-Echo':
        if mode=="train":
            content = content[:896]
        else:
            content = content[896:]
    else:
        print("Error: Specified dataset not found.")
        return 0

    return content

        
class EchoDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt

        # Specify CSV files
        if self.opt.dataset == 'TUS-Echo':
            csv_path_sub = opt.dataset_path + '/dataset_1_48k.csv'

            if self.opt.echo_fre == 'audible':
                csv_path = opt.dataset_path + '/dataset_1_48k.csv'
            elif self.opt.echo_fre == 'ultra':
                csv_path =  opt.dataset_path + '/dataset_20k_48k.csv'
            else:
                csv_path =  opt.dataset_path + '/dataset_20k_48k.csv'

        else:
            print("Error: Specified dataset not found.")
            return 0

        # main datas (ultrasonic echo)
        self.data_idx = parse_all_data(csv_path,self.opt.mode,self.opt.dataset)   

        # auxiliary datas (audible echo)
        self.data_idx_sub = parse_all_data(csv_path_sub,self.opt.mode,self.opt.dataset)   

        self.win_length = 64 
        self.base_audio_path = csv_path

    def __getitem__(self, index):
        """ Read main datas """
         
        if self.opt.dataset == 'TUS-Echo':
            audio_path = self.data_idx[index][0]
            audio_path = audio_path.replace('dataset', 'TUS-Echo')
            audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate, mono=False, duration=0.8)
                
            # cropping
            if self.opt.echo_fre == 'audible':
                start = int(audio_rate*0.07)
                end = int(audio_rate*0.8)
                audio = audio[:,start:end]
                audio = audio[:,start:]
            elif self.opt.echo_fre == 'ultra':
                start = int(audio_rate*0.1)
                end = int(audio_rate*0.7)
                audio = audio[:,start:end]
            else:
                start = int(audio_rate*0.07)
                end = int(audio_rate*0.8)
                audio = audio[:,start:end]

            # Converting echoes to spectrograms
            audio_spec_both = torch.FloatTensor(generate_spectrogram(audio[0,:], audio[1,:], audio[2,:], audio[3,:], self.win_length))

            if self.opt.echo_fre != '':
                audio_spec_both_sub = audio_spec_both
            else:
                """ Read auxiliary datas """
                audio_path_sub = self.data_idx_sub[index][0]
                audio_path_sub = audio_path_sub.replace('dataset', 'TUS-Echo')
                audio_sub, audio_rate_sub = librosa.load(audio_path_sub, sr=self.opt.audio_sampling_rate, mono=False, duration=0.8)
    
                # cropping
                start2 = int(audio_rate_sub*0.07)
                end2 = int(audio_rate_sub*0.8)
                audio_sub = audio_sub[:,start2:end2]
                audio_spec_both_sub = torch.FloatTensor(generate_spectrogram(audio_sub[0,:], audio_sub[1,:], audio_sub[2,:], audio_sub[3,:], self.win_length))


            """ Read Depth Maps """
            path = self.data_idx[index][1]
            path = path.replace('dataset', 'TUS-Echo')
            depth = np.load(path)

        else:
            print("Error: Specified dataset not found.")
            return 0

        # Processing Depth Maps
        depth = depth[:,80:560]
        depth = cv2.resize(depth, dsize=(128, 128))

        # Values exceeding 10000mm are treated as outliers
        border = 10000
        for i in range(128):
            for j in range(128):
                if depth[i][j]>border:
                    depth[i][j]=border

        depth = depth.reshape(1, 128, 128)
        depth = depth.astype(np.float32) # float32
        depth = torch.from_numpy(depth) # numpy => torch
 
        return {'depth':depth, 'audio':audio_spec_both, 'audio_sub':audio_spec_both_sub}
        # depth  : Depth Maps
        # audio  : Ultrasonic Echo Spectrogram
        # audio_sub : Audible Echo Spectrogram

    def __len__(self):
        return len(self.data_idx)

    def name(self):
        return 'EchoDataset'