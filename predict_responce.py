# Text Data Preprocessing Lib
import nltk

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import json
import pickle
import numpy as np
import random 

words=[] #list of unique roots words in the data
classes = [] #list of unique tags in the data
#list of the pair of (['words', 'of', 'the', 'sentence'], 'tags')
pattern_word_tags_list = [] 
ignore_words = ['?', '!',',','.', "'s", "'m"]

import tensorflow
from data_preprocessing import get_stem_words
model=tensorflow.keras.models.load_model("./chatbot_model.h5")

train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)
words=pickle.load(open('./words.pkl','rb'))
classes=pickle.load(open('./classes.pkl'))
def preprocess_user_input(user_input):
    input_word_token_1=nltk.word_tokenize(user_input)
    input_word_token_2=get_stem_words(input_word_token_1,ignore_words)
    input_word_token_2=sorted(list(set(input_word_token_2)))
    bag=[]
    bag_of_words=[]
    for word in words:            
            if word in input_word_token_2:              
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
    bag.append(bag_of_words)
    return np.array(bag)

def bot_class_predicition(user_input):
     inp=preprocess_user_input(user_input)
     prediction=model.predict(inp)
     predicted_class_label=np.argmax(prediction[0])
     return predicted_class_label
def bot_responce(user_input):
     predicted_class_label=bot_class_predicition(user_input)
     predicted_class=classes[predicted_class_label]
     for intend in intents['intents']:
          if intend ['tag']==predicted_class:
               bot_responce=random.choice(intend['responses'])
               return bot_responce
print('Hi I am Stella , How can I help you?') 
while True:
     user_input=input('type your msg')
     print("user input", user_input)
     responce=bot_responce(user_input)
     print("bot Responce",responce)         