from utils import listen_for_step, generate_flowchart
import speech_recognition as sr

def main():
    while True:
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            print("Waiting for the start phrase..")

            try:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=10)

                print("Recognizing...")
                start_phrase = recognizer.recognize_google(audio)

                if "start making the flowchart" in start_phrase.lower():
                    print("Starting flowchart maker...")
                    steps = []

                    while True:
                        step_text = listen_for_step()

                        if step_text == "flowchart completed":
                            break

                        steps.append(step_text)

                    generate_flowchart(steps)
                else:
                    print("Start phrase not detected. Exiting...")
            except sr.UnknownValueError:
                print("Could not understand audio. Please repeat.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            

if __name__ == "__main__":
    main()