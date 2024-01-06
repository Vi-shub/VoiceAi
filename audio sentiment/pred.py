from keras.utils import pad_sequences
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa
import os

model = load_model("./ser_best_5868.h5")
# emotion_dict = {
#     "01":"neutral",
#     "02":"calm",
#     "03":"happy",
#     "04":"sad",
#     "05":"angry",
#     "06":"fearful",
#     "07":"disgust",
#     "08":"surprised"
# }
emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

def get_features(path):
    mfcc = []
    zcr = []
    rmse = []


    all_features = []
    audio,s_r = librosa.load(path)

    mel_freq = np.mean(librosa.feature.mfcc(y = audio, sr = s_r, n_mfcc=50).T, axis = 0).reshape(-1)

    zero_cross = np.pad((librosa.feature.zero_crossing_rate(y = audio).reshape(-1)),
                            (0,(228-len(librosa.feature.zero_crossing_rate(y = audio).reshape(-1)))),
                            'constant',
                            constant_values=(0,0))

    rms = np.pad((librosa.feature.rms(y = audio).reshape(-1)),
                            (0,(228-len(librosa.feature.rms(y = audio).reshape(-1)))),
                            'constant',
                            constant_values=(0,0))
    
    
    mfcc.append(mel_freq)
    zcr.append(zero_cross)
    rmse.append(rms)

    all_features.append(np.hstack((mel_freq,zero_cross,rms)))

    return all_features

path = input("Enter file path: ")
x = np.array(get_features(path)).reshape(1,-1)

pred = emotions[np.argmax(model.predict(x))]
print(pred)