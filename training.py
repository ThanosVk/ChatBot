import nltk
from sklearn.model_selection import train_test_split
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,LSTM
from keras.optimizers import SGD
import random
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import matplotlib.pyplot as plt

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
intents = json.loads(open('intents.json').read())


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #Add documents in the corpus
        documents.append((w, intent['tag']))

        #Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


lemmatizer = WordNetLemmatizer()
#Lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower())
         for w in words
            if w not in ignore_words]
words = sorted(list(set(words)))
#Sort classes
classes = sorted(list(set(classes)))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#Create training data
training = []
#Create an empty array for our output
output_empty = [0] * len(classes)
#Training set, bag of words for each sentence
for doc in documents:
    #Initialize our bag of words
    bag = []
    #List of tokenized words for the pattern
    pattern_words = doc[0]
    #Lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    #Create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    #Output is a '0' for each tag and '1' for each pattern
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

#Shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

#Create train lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
x_train, x_test, y_train, y_test = train_test_split(train_x,train_y,test_size=0.1,random_state=5)
print("Training data created")
print(np.array(x_train).shape,np.array(y_train).shape,np.array(x_test).shape,np.array(y_test).shape)


#Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
#Equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(80, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Compile model with Stochastic gradient descent with Nesterov the best possible results
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Fitting and saving the model
hist = model.fit(x_train, y_train, epochs=200, batch_size=5, verbose=1,validation_data = (x_test, y_test))
model.save('model.h5', hist)

print("Model created")

test_results = model.evaluate(x_test,y_test, verbose=False)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}%')

plt.subplot(1,2,1)
plt.plot(hist.history['loss'],label="Loss",c='g')
plt.plot(hist.history['val_loss'],label="Validation Loss",c='c')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc='upper left')
plt.subplot(1,2,2)
plt.plot(hist.history['accuracy'],label="Accuracy",c='r')
plt.plot(hist.history['val_accuracy'],label="Validation Accuracy",c='b')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='upper left')
plt.show()

