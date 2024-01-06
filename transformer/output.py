from translate import translate
import re
print("\n")
def main():
    sentence = input("Enter text: ")
    sentence = re.sub(r'[^A-Za-z ]', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    result = ""
    for i in range (0,len(sentence),70):
        to_process = sentence[i:i+70]
        sub_output = translate(to_process)
        result = result + sub_output + " "
    result = re.sub(r"\s+"," ", result)
    print("\n")
    with open("summary.txt","w") as file:
        file.write(result)
    print(f"Summary: {result}")

main()