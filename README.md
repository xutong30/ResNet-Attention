# ResNet-Attention
Datasets:

WINGBEATS: https://www.kaggle.com/potamitis/wingbeats

ABUZZ: https://web.stanford.edu/group/prakash-lab/cgi-bin/mosquitofreq/the-science/data/

# DataGeneration.py:
Generating training data, validation data
Generating 3 csv files:
All csv files formatting: [data_name, genera, species]
csv 1: all data
csv 2: training data
csv 3: validation data
Will be used in Data_prepro.py and Train.py
# Data_prepro.py:
Data preprocessing: convert raw audio into mel-spectrograms
And load data into DataSet, so to use in Train.py.
To load data in DataSet: 2 ways are available
Way1: save spectrograms as images and then read from images - used this way when running in the ANU server
Way2: read data from spectrogram directly

# Model.py:
Our Model: combines ResNet-18 and Self-Attention Mechcnism
Will be used in Train.py
# Train.py:
Training the model
