from moviepy.video.io.VideoFileClip import VideoFileClip
import wave
from vosk import Model,KaldiRecognizer


def getTranscript(file):
    # check if the file is mp4
    if file.name.endswith('mp4'):
        video = VideoFileClip(file)
        audio = video.audio
        audio.write_audiofile('audio.wav')
    elif file.name.endswith('wav'):
        audio = wave.open(file)
        audio.writeframes('audio.wav')
    elif file.name.endswith('mp3'):
        audio = wave.open(file)
        audio.writeframes('audio.wav')
    else:
        return "File not supported"
    model = Model(r'.\vosk-model-en-in-0.5')
    wf = wave.open('audio.wav', 'rb')
    recognizer = KaldiRecognizer(model, wf.getframerate())
    recognizer.SetWords(True)
    text = ""
    while True:
        pass


