code:

from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Dropout, Activation

max_len = 100       # Maximum sequence length
n_words = 5000      # Vocabulary size
n_tags = 17         # Number of unique tags (classes)

input = Input(shape=(max_len,))

model = Embedding(input_dim=n_words, output_dim=64, input_length=max_len)(input)

model = Bidirectional(LSTM(units=64, return_sequences=True, recurrent_dropout=0.1))(model)

model = Dropout(0.1)(model)

output = TimeDistributed(Dense(n_tags, activation="softmax"))(model)

model = Model(inputs=input, outputs=output)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

output:

<img width="1071" height="312" alt="Screenshot 2025-10-08 110022" src="https://github.com/user-attachments/assets/7c017723-cdfe-4a24-aa42-ac7d4279955f" />
