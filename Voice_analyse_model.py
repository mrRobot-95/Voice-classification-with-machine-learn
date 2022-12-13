"""
author - N@vindu
Input - voice files from (https://drive.google.com/drive/folders/1HdE6trrwraLuS7pvPTrTg9aiBzSlJb-4?usp=sharing) save in to root file
"""

import librosa
import librosa.display
import pandas as pd
import numpy as np
from tqdm import tqdm
from os import walk
import pickle




mypath1="./voice/HEALTHY_SPEECH_COUNTING"
mypath2="./voice/POSITIVE_SPEECH_COUNTING"
mypath3="./voice/HEALTHY_COUGH"
mypath4="./voice/POSITIVE_COUGH"
mypath5="./voice/HEALTHY_SPEECH_VOWELS"
mypath6="./voice/POSITIVE_SPEECH_VOWELS"
mypath7="./voice/BREATHING_HEALTHY"
mypath8="./voice/BREATHING_POSITIVE"
mypath9="./voice/input"

# mypath1,mypath2,mypath3,mypath4,mypath5,mypath6,mypath7,mypath8
# if type of voices are high number of time will increases

x=[mypath1,mypath2,mypath3,mypath4,mypath5,mypath6,mypath7,mypath8]
f = []
for i in x:
    for (dirpath, dirnames, filenames) in walk(i):
        f.extend(filenames)
        break

print(f)

d={}
label=''
path=""
file=""

'''
Labeling the voice files according to voice type and save the file path into dictionary. Please change path according to files where saved. 
Read instructions for labeling methodes
'''

for i in f:
    if 'HSCN' in i:
        label=1
        path="./voice/HEALTHY_SPEECH_COUNTING"
        file=path+'/'+i
        d[i]=file,label
    elif 'HSCF' in i:
        label = 1
        path = "./voice/HEALTHY_SPEECH_COUNTING"
        file = path + '/' + i
        d[i] = file, label
    elif 'PSCN' in i:
        label = 2
        path = "./voice/POSITIVE_SPEECH_COUNTING"
        file = path + '/' + i
        d[i] = file, label
    elif 'PSCF' in i:
        label = 2
        path = "./voice/POSITIVE_SPEECH_COUNTING"
        file = path + '/' + i
        d[i] = file, label
    if 'HCH' in i:
        label = 1
        path = "./voice/HEALTHY_COUGH"
        file = path + '/' + i
        d[i] = file, label
    elif 'HCS' in i:
        label = 1
        path = "./voice/HEALTHY_COUGH"
        file = path + '/' + i
        d[i] = file, label
    elif 'PCH' in i:
        label = 2
        path = "./voice/POSITIVE_COUGH"
        file = path + '/' + i
        d[i] = file, label
    elif 'PCS' in i:
        label = 2
        path = "./voice/POSITIVE_COUGH"
        file = path + '/' + i
        d[i] = file, label
    elif 'HSVA' in i:
        label = 1
        path = "./voice/HEALTHY_SPEECH_VOWELS"
        file = path + '/' + i
        d[i] = file, label
    elif 'HSVE' in i:
        label = 2
        path = "./voice/HEALTHY_SPEECH_VOWELS"
        file = path + '/' + i
        d[i] = file, label
    elif 'HSVO' in i:
        label = 3
        path = "./voice/HEALTHY_SPEECH_VOWELS"
        file = path + '/' + i
        d[i] = file, label
    elif 'PSVA' in i:
        label = 4
        path = "./voice/POSITIVE_SPEECH_VOWELS"
        file = path + '/' + i
        d[i] = file, label
    elif 'PSVE' in i:
        label = 5
        path = "./voice/POSITIVE_SPEECH_VOWELS"
        file = path + '/' + i
        d[i] = file, label
    elif 'PSVO' in i:
        label = 6
        path = "./voice/POSITIVE_SPEECH_VOWELS"
        file = path + '/' + i
        d[i] = file, label
    if 'HBD' in i:
        label = 1
        path = "./voice/BREATHING_HEALTHY"
        file = path + '/' + i
        d[i] = file, label
    elif 'HBS' in i:
        label = 2
        path = "./voice/BREATHING_HEALTHY"
        file = path + '/' + i
        d[i] = file, label
    elif 'PBD' in i:
        label = 2
        path = "./voice/BREATHING_POSITIVE"
        file = path + '/' + i
        d[i] = file, label
    elif 'PBS' in i:
        label = 4
        path = "./voice/BREATHING_POSITIVE"
        file = path + '/' + i
        d[i] = file, label



df=pd.DataFrame.from_dict(d,orient='index',columns=['relative_path', 'classID'])

print(df)

from scipy.io import wavfile
import noisereduce as nr

def noice_reduce(file):

    # load data
    rate, data = wavfile.read(file_name)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    return reduced_noise



def features_extractor(file):
    #load the file (audio)
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

    #we extract mfcc
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
    #in order to find out scaled feature we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

#Now we ned to extract the featured from all the audio files so we use tqdm

### Now we iterate through every audio file and extract features
### using Mel-Frequency Cepstral Coefficients
i=1
extracted_features=[]
for index_num,row in tqdm(df.iterrows()):
    file_name = str(row["relative_path"])
    final_class_labels=row["classID"]
    # print(file_name)
    newdata=noice_reduce(file_name)
    data=features_extractor(newdata)
    extracted_features.append([data,final_class_labels])

### converting extracted_features to Pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
print(extracted_features_df.head())
# extracted_features_df.to_excel('features.xlsx')

### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
with open('x_breath_nr_q','wb') as f:
    pickle.dump(X, f)

y=np.array(extracted_features_df['class'].tolist())
with open('y_breath_nr_q','wb') as f:
    pickle.dump(y, f)



### Label Encoding -> Label Encoder


from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder



labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))


### Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Activation,Flatten

### No of classes
num_labels=y.shape[1]

model=Sequential()
###first layer
model.add(Dense(100,input_shape=(80,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

## Trianing model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from datetime import datetime
num_epochs = 1000
num_batch_size = 16
checkpointer = ModelCheckpoint(filepath='./audio_classification.hdf5',
                               verbose=1, save_best_only=True)
start = datetime.now()
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
duration = datetime.now() - start
print("Training completed in time: ", duration)

test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])


import tensorflow as tf
filename = 'model_breath_nr_q.h5'
tf.keras.models.save_model(model, filename)
#model.predict_classes(X_test)


predict_x=model.predict(X_test)
classes_x=np.argmax(predict_x,axis=1)
print(classes_x)
