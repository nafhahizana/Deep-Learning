code:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
# Assume 'corpus' contains cleaned Shakespeare text lines
tokenizer = Tokenizer()
# Assuming 'corpus' is a list of strings or a single string
# If corpus is not defined, you will need to load your text data here.
# Example: corpus = ["This is the first line.", "This is the second line."]
# Or: with open('shakespeare.txt', 'r') as f: corpus = f.readlines()
# Make sure 'corpus' is defined before this cell is run.
# For demonstration purposes, let's use a placeholder corpus:
corpus = ["This is a sample sentence for demonstration.", "Another sample sentence here."]

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
max_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')
X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = to_categorical(y, num_classes=total_words)
model = Sequential([
Embedding(total_words, 100, input_length=max_len-1),
LSTM(150),
Dense(total_words, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam',
metrics=['accuracy'])
model.fit(X, y, epochs=50, verbose=1)

output:

![Screenshot_17-9-2025_10937_colab research google com](https://github.com/user-attachments/assets/e146f80c-7a2e-43bb-89f6-c3fabfdf753c)
