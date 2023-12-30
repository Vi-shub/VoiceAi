from utils import *
import os

speakers = []

def pre_meeting():
    n = int(input("Enter the number of people in the meeting: "))
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



def main():
    if not os.path.exists("gmm_models"):
        os.makedirs("gmm_models")
    if not os.path.exists("Meet_Files"):
        os.makedirs("Meet_Files")
    with open("input.txt","w") as f:
        f.write("0")
    # pre_meeting()
    while_meeting()
    post_meeting()


if __name__ == "__main__":
    main()
    
