import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

datastore = []

with open('Sarcasm_Headlines_Dataset.json', 'r') as file:
    datastore = [json.loads(line) for line in file]

# sentences = [datastore['headline'] for item in datastore.items()]

sentences = [item['headline'] for item in datastore]
labels = [item['is_sarcastic'] for item in datastore]
urls = [item['article_link'] for item in datastore]

VOCAB_SIZE = 1000
EMB_DIM = 16
MAX_LEN = 120
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOK = "<OOV>"
TRAINING_SIZE = 10000

training_sentences = sentences[0:TRAINING_SIZE]
testing_sentences = sentences[TRAINING_SIZE:]

training_labels = labels[0:TRAINING_SIZE]
testing_labels = labels[TRAINING_SIZE:]

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

print(f"Word Index: {word_index} and Total Words: {len(tokenizer.word_index)}")

# convert the tokenized words into the sequence of integers for training data
training_sequences = tokenizer.texts_to_sequences(training_sentences)

# pad the sequences to match the sequence length for every sentence
training_padded_sequences = pad_sequences(
                training_sequences, maxlen=MAX_LEN, truncating=TRUNC_TYPE,
                padding=PADDING_TYPE,
)

print(f"Training Sequences: {training_sequences[0]} Training Padded Sequences: {training_padded_sequences[0]}")

# convert the tokenized words into the sequence of integers for testing data
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

# pad the sequences to match the sequence length for every sentence in testing data
testing_padded_sequences = pad_sequences(
                testing_sequences, maxlen=MAX_LEN, truncating=TRUNC_TYPE,
                padding=PADDING_TYPE,
)
print("=" * 50)
print(f"Testing Sequences: {testing_sequences[0]} Testing Padded Sequences: {testing_padded_sequences[0]}")

# Building the model
model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMB_DIM, input_length=MAX_LEN),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
])

# model summary
print(model.summary())

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
NUM_EPOCHS = 10

training_padded = np.array(training_padded_sequences)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded_sequences)
testing_labels = np.array(testing_labels)

history = model.fit(training_padded, training_labels, epochs=NUM_EPOCHS,
        validation_data=(testing_padded, testing_labels),verbose=1
)

# save the model results
model.save('test.h5')
