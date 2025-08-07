code:

from keras.models import Sequential 
from keras.layers import Dense 
import numpy as np 
import matplotlib.pyplot as plt 

X = np.array([[0,0], [0,1], [1,0], [1,1]]) 
Y = np.array([[0], [1], [1], [0]]) 

model = Sequential() 
model.add(Dense(4, input_dim=2, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
history = model.fit(X, Y, epochs=1000, verbose=0) 
loss, acc = model.evaluate(X, Y) 

print("Accuracy:", acc) 
print("Predictions:", model.predict(X)) 
plt.plot(history.history['loss']) 
plt.title('Model Loss') 
plt.xlabel('Epoch') 
plt.ylabel('Loss') 
plt.grid(True) 
plt.show()


output:

![Deep learning exp 1](https://github.com/user-attachments/assets/1b54df69-b1ae-4edb-8dd0-17d47af84485)
