import pandas as pd
import librosa
import numpy as np
import librosa.display
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


import os
import random

# data augmentation
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
    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v - f)
        mel_spectrogram[:, f0:f0 + f, :] = 0

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau - t)
        mel_spectrogram[:, :, t0:t0 + t] = 0
    return mel_spectrogram



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

        spec = Image.open(audio_path)
        spec = self.transform(spec)
        spec = specaug(spec)

        # label get from self.label
        aname = audio_name
        label_class = self.label[self.label['Fname'] == aname[:-4]].iloc[0]['Species']
        label = torch.tensor(self.label_list[label_class])
        return spec, label

    def __len__(self):
        #         length of the whole dataset
        return len(self.data_path)