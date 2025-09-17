code:

from keras.models import Sequential 
from keras.layers import Embedding, GRU, Dense 
model = Sequential([ 
Embedding(10000, 32, input_length=100), 
GRU(100), 
Dense(1, activation='sigmoid') 
]) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

output:

![Screenshot_17-9-2025_115034_colab research google com](https://github.com/user-attachments/assets/762b6eaa-3a35-4349-bc2c-acc2d2d57232)
