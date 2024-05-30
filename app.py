import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

import streamlit as st
import numpy as np

#loading the pre-trained weights and model architecture
model = tf.keras.models.load_model("./model/next-word-model.h5")

st.title("Get next word!")
st.subheader("A Deep Learning Model that predicts the next likely sequence of words")

file = open("./corpus/goodwill.txt").read() 

tokenizer = Tokenizer() 
data = file.lower().split("\n") 

corpus = []
for line in data:
    a = line.strip()
    corpus.append(a)

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print(tokenizer.word_index)
print(total_words)

# Creating labels for each sentence in dataset
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# Padding the sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

num = st.slider("Number of text predictions?",0,10)

# Generating next words given a seed sentence
def next_word(seed):
  seed_text = seed
  next_words = num
  for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    predicted = np.argmax(model.predict(token_list), axis=1)

    output_word = ""
    for word, index in tokenizer.word_index.items():
      if index == predicted:
        output_word = word
        break
    seed_text += " " + output_word
  st.subheader(seed_text)

# Getting the output/predicted text  
next_word(st.text_input('Enter seed sentence', 'I am going to'))