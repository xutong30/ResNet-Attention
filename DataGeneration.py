import os
import shutil
import pandas as pd
import csv

# make lists of data for the dataset
# eg: wingbeats dataset in this code
# containing both all data in the dataset
# format: file + type + generna
with open('wing_labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    path = './WINGBEATS/'
    Ae_aegypti = 'Ae. aegypti'
    Ae_albopictus = 'Ae. albopictus'
    An_arabiensis = 'An. arabiensis'
    An_gambiae = 'An. gambiae'
    C_pipiens = 'C. pipiens'
    C_quinquefasciatus = 'C. quinquefasciatus'
    aegypti = os.listdir(path + Ae_aegypti)
    al = os.listdir(path + Ae_albopictus)
    ar = os.listdir(path + An_arabiensis)
    ga = os.listdir(path + An_gambiae)
    pi = os.listdir(path + C_pipiens)
    qu = os.listdir(path + C_quinquefasciatus)
    count = 0
    for fname in aegypti:
        name = fname[0:-4]
        writer.writerow([name, "Ae", "Aegypti"])
        count += 1
    for fname in al:
        name = fname[0:-4]
        writer.writerow([name, "Ae", "albopictus"])
        count += 1
    for fname in ar:
        name = fname[0:-4]
        writer.writerow([name, "An", "arabiensis"])
        count += 1
    for fname in ga:
        name = fname[0:-4]
        writer.writerow([name, "An", "gambiae"])
        count += 1
    for fname in pi:
        name = fname[0:-4]
        writer.writerow([name, "C", "pipiens"])
        count += 1
    for fname in qu:
        name = fname[0:-4]
        writer.writerow([name, "C", "quinquefasciatus"])
        count += 1


# calculate the number of samples in each species
path = './WINGBEATS/'
Ae_aegypti = 'Ae. aegypti'
Ae_albopictus = 'Ae. albopictus'
An_arabiensis = 'An. arabiensis'
An_gambiae = 'An. gambiae'
C_pipiens = 'C. pipiens'
C_quinquefasciatus = 'C. quinquefasciatus'

ae = os.listdir(path+Ae_aegypti)
al = os.listdir(path+Ae_albopictus)
ar = os.listdir(path+An_arabiensis)
ga = os.listdir(path+An_gambiae)
pi = os.listdir(path+C_pipiens)
qu = os.listdir(path+C_quinquefasciatus)

ae_count = 0
al_count = 0
ar_count = 0
ga_count = 0
pi_count = 0
qu_count = 0
for fname in ae:
    ae_count += 1
for fname in al:
    al_count += 1
for fname in ar:
    ar_count += 1
for fname in ga:
    ga_count += 1
for fname in pi:
    pi_count += 1
for fname in qu:
    qu_count += 1


# separate data as training data and validation data
# 2 folders in total -> 1 folder: training data, 1 folder: validation data
aecount = 0
alcount = 0
arcount = 0
gacount = 0
picount = 0
qucount = 0
path = './WINGBEATS/'
train_path = './wing_train/'
vali_path = './wing_vali/'

print('ae...')
ae_train = 0
ae_vali = 0
for fname in ae:
    # for each species-> 80% in training data folder, 20% in validation data folder
    if aecount < ae_count * 0.8:
        shutil.copy(path + Ae_aegypti + '//' + fname, train_path + fname)
        ae_train += 1
    else:
        shutil.copy(path + Ae_aegypti + '//' + fname, vali_path + fname)
        ae_vali += 1
    aecount += 1
print('train: ', ae_train)
print('vali: ', ae_vali)
print('---------------------')

print('al...')
train = 0
vali = 0
for fname in al:
    if alcount < al_count * 0.8:
        shutil.copy(path + Ae_albopictus + '//' + fname, train_path + fname)
        train += 1
    else:
        shutil.copy(path + Ae_albopictus + '//' + fname, vali_path + fname)
        vali += 1
    alcount += 1
print('train: ', train)
print('vali: ', vali)
print('---------------------')

print('ar...')
train = 0
vali = 0
for fname in ar:
    if arcount < ar_count * 0.8:
        train += 1
        shutil.copy(path + An_arabiensis + '//' + fname, train_path + fname)
    else:
        shutil.copy(path + An_arabiensis + '//' + fname, vali_path + fname)
        vali += 1
    arcount += 1
print('train: ', train)
print('vali: ', vali)
print('---------------------')

print('ga...')
train = 0
vali = 0
for fname in ga:
    if gacount < ga_count * 0.8:
        shutil.copy(path + An_gambiae + '//' + fname, train_path + fname)
        train += 1
    else:
        shutil.copy(path + An_gambiae + '//' + fname, vali_path + fname)
        vali += 1
    gacount += 1
print('train: ', train)
print('vali: ', vali)
print('---------------------')

print('pi...')
train = 0
vali = 0
for fname in pi:
    if picount < pi_count * 0.8:
        shutil.copy(path + C_pipiens + '//' + fname, train_path + fname)
        train += 1
    else:
        shutil.copy(path + C_pipiens + '//' + fname, vali_path + fname)
        vali += 1
    picount += 1
print('train: ', train)
print('vali: ', vali)
print('---------------------')

print('qu...')
train = 0
vali = 0
for fname in qu:
    if qucount < qu_count * 0.8:
        shutil.copy(path + C_quinquefasciatus + '//' + fname, train_path + fname)
        train += 1
    else:
        shutil.copy(path + C_quinquefasciatus + '//' + fname, vali_path + fname)
        vali += 1
    qucount += 1
print('train: ', train)
print('vali: ', vali)
print('---------------------')

# read the data information into csv files - 2 csv files
# csv 1: recording training data, csv 2: recording validation data
# recoding format for each data: data name, genera, species

# read the csv file that contains all data
audios = pd.read_csv('./wing_labels.csv')

# generate the csv file for training data
train_file = os.listdir('./wing_train')
with open('wing_train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for fname in train_file:
        name = fname[0:-4]
        temp = []
        temp.append(audios[audios['Fname'] == name].iloc[0]['Fname'])
        temp.append(audios[audios['Fname'] == name].iloc[0]['Genera'])
        temp.append(audios[audios['Fname'] == name].iloc[0]['Species'])
        writer.writerow(temp)
        count += 1


# generate the csv file for validation data
vali_file = os.listdir('./wing_vali')
with open('wing_vali.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for fname in vali_file:
        if fname[-4:] != '.wav':
            continue
        name = fname[0:-4]
        temp = []
        temp.append(audios[audios['Fname'] == name].iloc[0]['Fname'])
        temp.append(audios[audios['Fname'] == name].iloc[0]['Genera'])
        temp.append(audios[audios['Fname'] == name].iloc[0]['Species'])
        writer.writerow(temp)
        count += 1

# add headers to our generated csv files
audios = pd.read_csv('wing_train.csv', header=None)
audios.to_csv("wing_train.csv", header=["Fname", "Genera", "Species"], index=False)
print(audios)

audios = pd.read_csv('wing_vali.csv', header=None)
audios.to_csv("wing_vali.csv", header=["Fname", "Genera", "Species"], index=False)
print(audios)
