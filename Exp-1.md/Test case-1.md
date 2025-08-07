code:

import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

model Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense (1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=1000, verbose=0)

predictions model.predict(X)
print("Predictions:", np.round(predictions).astype(int))

output:

![Deep learning test case 1](https://github.com/user-attachments/assets/5a508aa2-364a-4890-94f3-4333de555814)
