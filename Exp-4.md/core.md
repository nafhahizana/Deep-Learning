code:

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np
# Sample corpus
data = "Deep learning is amazing. Deep learning builds intelligent systems."
# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
sequences = []
words = data.split()
for i in range(1, len(words)):
    seq = words[:i+1]
    sequences.append(' '.join(seq))
# Integer encoding
encoded = tokenizer.texts_to_sequences(sequences)
max_len = max([len(x) for x in encoded])
X = np.array([x[:-1] for x in pad_sequences(encoded, maxlen=max_len)])
y = to_categorical([x[-1] for x in pad_sequences(encoded, maxlen=max_len)],
num_classes=len(tokenizer.word_index)+1)
# Model
model = Sequential([
Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=10,
input_length=max_len-1),
SimpleRNN(50),
Dense(len(tokenizer.word_index)+1, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam',
metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=0) 

output:

![Screenshot_17-9-2025_10517_colab research google com](https://github.com/user-attachments/assets/59570e24-a9a4-45cb-a967-3172ca19aaef)
