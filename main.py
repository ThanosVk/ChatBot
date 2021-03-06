import nltk
from nltk.stem import WordNetLemmatizer
import json
import random
import pickle
import numpy as np
from keras.models import load_model
from tkinter import *
lemmatizer = WordNetLemmatizer()

model = load_model('model.h5')

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    #Τokenize the pattern and split words into array
    sentence_words = nltk.word_tokenize(sentence)
    #Stem each word and create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
    #Return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    #Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    #Bag of words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                #Assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    #Filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    #Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


#Creating GUI with tkinter
#Send function to send the messages of the user to ChatBot
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

#Sending the message with the Enter key
def send_withEnter(event):
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

#Creation of the basic window
base = Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create chat area
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial", )

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

photo = PhotoImage(file = r"send.png")
#Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), image=photo, width="12", height=5,
                    bd=0, bg="#32c7de", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
EntryBox.bind("<Return>", send_withEnter)


#Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=265)
SendButton.place(x=265, y=401, height=90,width=115)

#Creation of secondary window for displaying information
def about_window():
    top=Toplevel()
    top.title("About")
    top.geometry("350x150")
    label = Label(top, text="This is a ChatBot created\n for the semester project.\n\nAuthors: Thanassis Vakouftsis,Alexandra Tsarouchi\nCopyright© 2021\n\n")
    label.pack()
    btn = Button(top,text="Close window",command=top.destroy).pack()
    top.resizable(False, False)
#Creation of window for exiting the ChatBot
def exit():
    top=Toplevel()
    top.title("Exit")
    top.geometry("350x150")
    label = Label(top, text="\n\nAre you sure you want to close the Chatbot?\n\n")
    label.pack()
    btn = Button(top, text="Yes", command=base.quit).pack(side=LEFT,padx=65)
    btn2 = Button(top,text="No",command=top.destroy).pack(side=RIGHT,padx=65)
    top.resizable(False, False)

#Menu Components
menubar = Menu(base)
filemenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="About", command=about_window)
menubar.add_cascade(label="Exit",command=exit)

base.config(menu=menubar)
base.mainloop()

