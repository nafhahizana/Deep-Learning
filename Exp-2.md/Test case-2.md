code:

import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
fron keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import_to categorical

(X_train, y_train), (X_test, y_test) fashion_mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense (128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer="adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_cat, epochs=5, batch_size=32, validation_data=(X_test, y_test_cat))


sample_indices = [0, 1, 2, 7]
predicted_labels = model.predict(X_test[sample_indices])
predicted_classes = np.argmax(predicted_labels, axis=1)


true_labels = y_test[sample_indices]
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker", 'Bag', 'Ankle boot']

print("Test Cases:")
print("Input Image\tTrue Label\tPredicted Label\tCorrect (Y/N)")
for i, idx in enumerate(sample_indices):
    true = class_names[true_labels[i]]
    pred = class_names[predicted_classes[i]]
    correct = 'Y' if true == pred else 'N'
    print (f"{true}\t{true}\t{pred}\t{correct}")

output:

![Deep Learning exp2 test case 2 2025-08-13 at 11 21 05_1704e744](https://github.com/user-attachments/assets/7815245b-753e-4089-a463-085a89c6c8dc)
