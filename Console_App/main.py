from utils import *
import os
from text_to_num import text2num
speakers = []
text = []
def pre_meeting():
    speak("Welcome to the Exec-u-Talk meeting assistant")
    speak("Please enter the number of people in the meeting")
    n = int(input("Enter the number of people in the meeting: "))
    print(n)
    names = []
    for i in range(n):
        name = record_audio_train()
        names.append(name)
    for name in names:
        train_model(name)
def while_meeting(i=1):
    i = 1
    print("Meeting started")
    while True:
        f = open("input.txt","r")
        if f.read() == "1":
            f.close()
            break
        else:
            f.close()
            speaker = record_and_predict_speaker(i)
            speakers.append(speaker)
            i += 1
def post_meeting():
    create_transcript(speakers)
    translateFile()
def main():
    if not os.path.exists("gmm_models"):
        os.makedirs("gmm_models")
    if not os.path.exists("Meet_Files"):
        os.makedirs("Meet_Files")
    with open("input.txt","w") as f:
        f.write("0")
    pre_meeting()
    while_meeting()
    with open("Transcript.txt","w") as out:
        out.write("")
    post_meeting()

if __name__ == "__main__":
    main()
    
