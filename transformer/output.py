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

# Addressing a rally in poll-bound Telangana, Uttar Pradesh Chief Minister Yogi Adityanath said that BJP will rename 'Karimnagar' as 'Karipuram' if voted to power in the December 7 Assembly polls. BJP would work towards the sentiments of the people of the state, he added. He also said that the Congress, TRS and TDP work for the appeasement of the Muslims.

#Madhesi Morcha, an alliance of seven political parties, has withdrawn support to PM Pushpa Kamal Dahal-led Nepal government after it failed to meet a seven-day ultimatum to fulfil their demands including endorsement for the revised Constitution amendment bill. The Morcha has 36 seats in the Parliament, but despite the withdrawal of support, there is no immediate threat to the government.