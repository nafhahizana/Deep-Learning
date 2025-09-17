code:

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np
text = "To be or not to be that is the question What light through yonder window breaks"
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
word_index = tokenizer.word_index
total_words = len(word_index) + 1
words = text.split()
sequences = []
for i in range(1, len(words)):
    seq = words[:i+1]
    sequences.append(' '.join(seq))
encoded = tokenizer.texts_to_sequences(sequences)
max_len = max(len(seq) for seq in encoded)
X = np.array([seq[:-1] for seq in pad_sequences(encoded, maxlen=max_len)])
y = to_categorical([seq[-1] for seq in pad_sequences(encoded, maxlen=max_len)], num_classes=total_words)
model = Sequential([
    Embedding(total_words, 10, input_length=max_len-1),
    SimpleRNN(50),
    Dense(total_words, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=0)
def predict_next_word(seed):
    token_list = tokenizer.texts_to_sequences([seed])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1)
    predicted = model.predict(token_list, verbose=0)
    index = np.argmax(predicted)
    for word, i in word_index.items():
        if i == index:
            return word
tests = ["To be or not", "What light through yonder window"]
for t in tests:
    print(f"Input: '{t}' → Next word: '{predict_next_word(t)}'")

  output:

  ![Screenshot_17-9-2025_112815_colab research google com](https://github.com/user-attachments/assets/a18fc628-f034-4753-94ce-9ec1874f5970)
