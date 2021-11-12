import pandas as pd
import librosa
import numpy as np
import librosa.display
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import skimage.io

import os
import random


# transform data from raw audio to spectrogram
def get_melspectrogram_db(file_path, sr=8000, top_db=80, dataset=None):

    wav, sr = librosa.load(file_path, sr=sr)
    if dataset == 'wingbeats':
        # padding the data in wingbeats data file as length of data in wingbeats is too short
        if wav.shape[0] < 2*sr:
            wav = np.pad(wav, int(np.ceil((2*sr-wav.shape[0])/2)), mode='reflect')
        else:
            wav = wav[:2*sr]
    # get spectrogram
    spec = librosa.feature.melspectrogram(wav, sr=sr, n_fft=256, hop_length=64)
    # transform to decibel based spectrogram
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db


# normalize the spectrogram
def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


# data augmentation on time domain and frequency domain in spectrogram image - have 3 channels
def specaug(mel_spectrogram, frequency_masking_para=10,
            time_masking_para=10, frequency_mask_num=1, time_mask_num=1):
    """
        Modified from SpecAugment
        Author: Demis TaeKyu Eom and Evangelos Kazakos
        License: https://github.com/DemisEom/SpecAugment/blob/master/LICENSE
        Code URL: https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_pytorch.py
    """
    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[2]
    # Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v - f)
        mel_spectrogram[:, f0:f0 + f, :] = 0

    # Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau - t)
        mel_spectrogram[:, :, t0:t0 + t] = 0
    return mel_spectrogram


# data augmentation on time domain and frequency domain in spectrogram - have 2 channels
def specaug_spec(mel_spectrogram, frequency_masking_para=10, time_masking_para=10, frequency_mask_num=1, time_mask_num=1):

    v = mel_spectrogram.shape[0]
    tau = mel_spectrogram.shape[1]
    # Step 1 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v - f)
        mel_spectrogram[f0:f0+f, :] = 0

    # Step 2 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau - t)
        mel_spectrogram[:, t0:t0+t] = 0
    return mel_spectrogram


# normalize spectrogem and save as png in to the folder
def spec_to_image_2(spec, filename, out, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)

    output = out + filename + '.png'
    # save as PNG
    skimage.io.imsave(output, spec_scaled)


# save spectrograms as images in folders
# train_d, valid_d: csv files of training and validation data
# train_path, vali_path: training and validation data audio folder
# train_img, vali_img: training and validation image folders which is to store spectrograms as images.
def save_spec_to_img(train_d, valid_d, train_path, vali_path, train_img, vali_img):
    for i in range(len(train_d)):
        filename = train_d.loc[i].at["Fname"]
        file_path = train_path + filename + '.wav'
        spec_to_image_2(get_melspectrogram_db(file_path), filename, train_img)

    for i in range(len(valid_d)):
        filename = valid_d.loc[i].at["Fname"]
        file_path = vali_path + filename + '.wav'
        spec_to_image_2(get_melspectrogram_db(file_path), filename, vali_img)


# Store data into Dataset
class ListDataset(Dataset):
    def __init__(self, data_path, label_file, label_list):
        self.data = data_path
        self.label = pd.read_csv(label_file)
        self.data_path = os.listdir(self.data)
        self.label_list = label_list
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        audio_name = self.data_path[index]
        audio_path = os.path.join(self.data, audio_name)

        # way 1: get spectrogram from the image that related to a specific index
        # spec = Image.open(audio_path)
        # spec = self.transform(spec)
        # spec = specaug(spec)

        # way 2: get spectrogram from the raw audio sounds that related to a specific index
        audio_name = self.data_path[index]
        audio_path = os.path.join(self.data, audio_name)
        # get spectrogram from raw audio
        spec = spec_to_image(get_melspectrogram_db(audio_path))[np.newaxis, ...]
        spec = specaug_spec(spec)

        # get its label
        aname = audio_name
        label_class = self.label[self.label['Fname'] == aname[:-4]].iloc[0]['Species']
        label = torch.tensor(self.label_list[label_class])

        return spec, label

    def __len__(self):
        #         length of the whole dataset
        return len(self.data_path)