import pyaudio
import wave
import os
import numpy as np
from python_speech_features import mfcc
from sklearn import preprocessing
import librosa
from sklearn.mixture import GaussianMixture
import pickle
import json
from vosk import Model,KaldiRecognizer
import speech_recognition as sr
from graphviz import Digraph
from easygoogletranslate import EasyGoogleTranslate
import pyttsx3
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize
from transformer.translate import translate as summary_translate
import re
sample_paragraphs = {
    1: "The average human heart beats about 100,000 times a day.And is the most important organ in the body.",
    
    2: "A single day on Venus is longer than a year on Earth, lasting about 243 Earth days. Venus is the second brightest natural object in the sky.",
    
    3: "The Eiffel Tower in Paris is made up of over 18,000 individual iron parts. The Eiffel Tower was named after the engineer Gustave Eiffel.",
    
    4: "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 63 million square miles. The Pacific Ocean has an average depth of 2.4 miles.",
    
    5: "Bananas are berries, but strawberries are not. In botanical terms, berries are defined by certain criteria, and bananas fit the bill"
}


def record_audio_train():
    speak("Please enter your name...")
    name = input("Please enter your name: ")
    print("Your name is "+name)
    os.mkdir(f"traning_data_{name}")
    speak("Recording your voice samples...Please speak the following sentences written on the screen")
    for count in range(5):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 10
        device_index = 2
        audio = pyaudio.PyAudio()
        index = 1
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,input_device_index = index,
                            frames_per_buffer=CHUNK)
        print(sample_paragraphs[count+1])
        print ("recording started")
        Recordframes = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)
        print ("recording stopped")
        stream.start_stream()
        stream.close()
        audio.terminate()
        OUTPUT_FILENAME = name+"-sample"+str(count)+".wav"
        WAVE_OUTPUT_FILENAME = os.path.join(f"traning_data_{name}",OUTPUT_FILENAME)
        trainedfilelist = open(f"trainedfilelist{name}.txt","a")
        trainedfilelist.write(OUTPUT_FILENAME+"\n")
        wavefile = wave.open(f"traning_data_{name}/{OUTPUT_FILENAME}", 'wb')
        wavefile.setnchannels(CHANNELS)
        wavefile.setsampwidth(audio.get_sample_size(FORMAT))
        wavefile.setframerate(RATE)
        wavefile.writeframes(b''.join(Recordframes))
        wavefile.close()
    return name

    
def extract_features(audio,rate):
        mfcc_feature = mfcc(audio,rate,0.025,0.01,20,nfft = 1200,appendEnergy = True)
        mfcc_feature = preprocessing.scale(mfcc_feature)
        delta = librosa.feature.delta(mfcc_feature)
        combined = np.hstack((mfcc_feature,delta))
        return combined
    
def train_model(name):
        file_paths = open(f"trainedfilelist{name}.txt","r")
        count = 1
        features = np.asarray(())
        for path in file_paths:
            path = path.strip()
            print(path)
            audio,sr = librosa.load(os.path.join(f"traning_data_{name}",path),sr = None)
            vector = extract_features(audio,sr)
            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features,vector))

            
            if count == 5:
                gmm = GaussianMixture(n_components=6,max_iter = 200,covariance_type='diag',n_init = 3)
                gmm.fit(features)
                picklefile = path.split("-")[0]+".gmm"
                pickle.dump(gmm,open(os.path.join("gmm_models",picklefile),'wb'))
                print ('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
                features = np.asarray(())
                count = 0
            count = count + 1

def record_and_predict_speaker(j):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 10
        device_index = 2
        audio = pyaudio.PyAudio()
        index = 1
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,input_device_index = index,
                            frames_per_buffer=CHUNK)
        Recordframes = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)
        stream.start_stream()
        stream.close()
        audio.terminate()
        OUTPUT_FILENAME = f"chunk{j}.wav"
        WAVE_OUTPUT_FILENAME = os.path.join("Meet_Files",OUTPUT_FILENAME)
        wavefile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wavefile.setnchannels(CHANNELS)
        wavefile.setsampwidth(audio.get_sample_size(FORMAT))
        wavefile.setframerate(RATE)
        wavefile.writeframes(b''.join(Recordframes))
        wavefile.close()
        audio,sr = librosa.load(WAVE_OUTPUT_FILENAME,sr = None)
        vector = extract_features(audio,sr)
        gmm_files = [os.path.join("gmm_models",fname) for fname in os.listdir("gmm_models") if fname.endswith('.gmm')]
        models = [pickle.load(open(fname,'rb')) for fname in gmm_files]
        speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm = models[i]
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        return speakers[winner]

def create_transcript(speakers):
    model = Model(r'.\vosk-model-en-in-0.5')
    i = 1
    flag = 0

    for files in os.listdir("Meet_files"):
         if i== len(speakers)+1:
                break
         wf = wave.open(f"Meet_Files/chunk{i}.wav", "rb")
         recognizer = KaldiRecognizer(model, wf.getframerate())
         recognizer.SetWords(True)
         textresults = []
         results = ""
         while True:
              data = wf.readframes(4000)
              if len(data) == 0:
                  break
              if recognizer.AcceptWaveform(data):
                   recognizerResult = recognizer.Result()
                   results = results + recognizerResult
                   resultDict = json.loads(recognizerResult)
                   textresults.append(resultDict.get("text", ""))
         ressultDict = json.loads(recognizer.FinalResult())
         textresults.append(ressultDict.get("text", ""))
         print(textresults)
         with open("Transcript.txt","a") as out:
              out.write(f"{speakers[i-1][11:]}: ")
              for result in textresults:
                   out.write(result)
              out.write("\n")
         i+=1         
def listen_for_step():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for the next step...")

        try:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=10)

            print("Recognizing...")
            step_text = recognizer.recognize_google(audio)
            print(f"Step: {step_text}")

            return step_text

        except sr.UnknownValueError:
            print("Could not understand audio. Please repeat.")
            return listen_for_step()
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return listen_for_step()
        
def generate_flowchart(steps):
    dot = Digraph(comment="Flowchart")

    for index, step_text in enumerate(steps, start=1):
        dot.node(f"step_{index}", step_text)

    for i in range(1, len(steps)):
        dot.edge(f"step_{i}", f"step_{i+1}", label=f"Step {i}")

    dot.render(f"{steps[0]}", format="png", cleanup=True)

    print("Flowchart saved as flowchart.png")


def translate(text,lang):
    chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
    translator = EasyGoogleTranslate(
          source_language = "en",
          target_language = lang
    )
    translated_text = ""
    for chunk in chunks:
          translated_text += translator.translate(chunk)
    return translated_text

def translateFile():
    langs = ["hi","mr","ta","te","kn"]
    print(1)
    for lang in langs:
        print(2)
        with open("Transcript.txt","r") as file:
            print(3)
            text = file.read()
            print(4)
        with open(f"Transcript_{lang}.txt","w",encoding='utf8') as file:
            print(5)
            file.write(translate(text,lang))
            print(6)
        print(f"Translated to {lang}")


def record_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=10)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Could not understand audio")
        return record_input()

def speak(text):
    print(text)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
    

def sentimenAnalysis():
    with open("Transcript.txt","r") as file:
        text = file.read()
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    speak("The meeting rated "+str(sentiment["pos"]*100)+"% positive, "+str(sentiment["neg"]*100)+"% negative and "+str(sentiment["neu"]*100)+"% neutral")
    speak("The summary of the meeting is saved in summary dot t.x.t. Thank you for using Exec-u-Talk")

def summarize():
    with open("Transcript.txt","r") as file:
        lines = file.readlines()
    extracted_sentences = []
    for line in lines:
        extracted_sentences.append(line.split(":")[1])
    text = "".join(extracted_sentences)
    text = re.sub(r'[^A-Za-z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    result = ""
    for i in range (0,len(text),70):
        to_process = text[i:i+70]
        sub_output = summary_translate(to_process)
        result = result + sub_output + " "
    result = re.sub(r"\s+"," ", result)
    with open("summary.txt","w") as file:
        file.write(result)
    speak(f"The summary of the meeting is {result}")
    speak("The summary is saved in summary dot t.x.t. Thank you for using Exec-u-Talk")




    
    
    




         