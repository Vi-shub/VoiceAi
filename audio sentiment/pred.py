from keras.utils import pad_sequences
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa
import os

def ser(path):
    model = load_model("./ser_best_5868.h5")
    emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    prediction_list = []

    def get_features(audio, s_r):
        mfcc = []
        zcr = []
        rmse = []

        # Extract features
        mel_freq = np.mean(librosa.feature.mfcc(y=audio, sr=s_r, n_mfcc=50).T, axis=0).reshape(-1)
        zero_cross = np.pad(librosa.feature.zero_crossing_rate(y=audio).reshape(-1), (0, 228 - len(librosa.feature.zero_crossing_rate(y=audio).reshape(-1))), 'constant')
        rms = np.pad(librosa.feature.rms(y=audio).reshape(-1), (0, 228 - len(librosa.feature.rms(y=audio).reshape(-1))), 'constant')

        mfcc.append(mel_freq)
        zcr.append(zero_cross)
        rmse.append(rms)

        all_features = np.hstack((mel_freq, zero_cross, rms)).reshape(1, -1)

        return all_features

    audio, s_r = librosa.load(path)

    # chunk duration 2 seconds
    chunk_duration = 3  
    chunk_samples = int(chunk_duration * s_r)
    chunks = [audio[i:i + chunk_samples] for i in range(0, len(audio), chunk_samples)]

    for i, chunk in enumerate(chunks):
        x = np.array(get_features(chunk, s_r))
        pred = emotions[np.argmax(model.predict(x))]
        prediction_list.append(pred)

    return prediction_list

print(ser(r"C:\Users\Rommel\OneDrive\Documents\Sound Recordings\Recording (4).wav"))
