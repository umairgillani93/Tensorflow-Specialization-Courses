import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, Conv1D, Bidirectional, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import Regularizer
import tensorflow.keras.utils as ku

with open(file=os.getcwd() + '/text.txt') as file:
    result = file.read()

print(result)

corpus = result.lower().split('\n')
print("The corpus is: \n", corpus)

# tokenize the words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
total_words = len(tokenizer.word_index) +1
print("Word Index: ", word_index)
print("\nTotal Words: ", total_words)

# making sequences of integers from tokenized words
input_sequence = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequence.append(n_gram_sequence)

# padding the sequences
max_len = max(len(x) for x in input_sequence)
input_sequence = np.array(pad_sequences(input_sequence, maxlen=max_len, padding='pre'))

print(max_len)
print(input_sequence)

# creating predictors and labels
predictors, labels = input_sequence[:,:-1], input_sequence[:-1]
labels = ku.to_categorical(labels, num_classes=total_words)

# building the model
VOCAB_SIZE = total_words

model = Sequential()
model.add(Embedding(VOCAB_SIZE, 64, input_length=max_len-1))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.25))
model.add(LSTM(32))
model.add(Dense(total_words/2, activation='relu'))
model.add(Dense(total_words, activation='softmax'))
# compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
NUM_EPOCHS = 10

history = model.fit(predictors, labels, epochs=NUM_EPOCHS, verbose=1)
