import numpy as np
from pathlib import Path
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

import matplotlib.pyplot as plt

train_path = Path('./imdb/train')
test_path = Path('./imdb/test')

# Truncate reviews to 100 words
max_len = 100
# Use 10 000 most common words
max_words = 10000

labels = []
texts = []

for label, child_dir in enumerate(['neg', 'pos']):
    for text in (train_path / child_dir).glob('*.txt'):
        texts.append(text.read_text())
        labels.append(label)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

print(f'found {len(word_index)} unique tokens')

data = pad_sequences(sequences, maxlen=max_len)
labels = np.array(labels)

# Shuffle data
shuffel_index = np.random.permutation(len(labels))
x = data[shuffel_index]
y = labels[shuffel_index]

print('data shape:', x.shape)
print('label shape:', y.shape)


def load_embeddings():
    embedding_index = {}
    glove = Path('./glove/glove.6B.100d.txt')
    with glove.open() as f:
        for line in f:
            values = line.split()
            word = values[0]
            coef = np.array(values[1:], dtype='float32')
            embedding_index[word] = coef

    print(f'found {len(embedding_index)} word vectors')

    return embedding_index


embedding_index = load_embeddings()
embdding_dim = 100


def build_embdding_matrix():
    embdding_matrix = np.zeros((max_words, embdding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embdding_matrix[i] = embedding_vector

    return embdding_matrix


model = Sequential([
    Embedding(max_words, embdding_dim, input_length=max_len),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

print(model.summary())

model.layers[0].set_weights([build_embdding_matrix()])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x,
                    y,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

model.save_weights('pre_trained_glove_model.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
