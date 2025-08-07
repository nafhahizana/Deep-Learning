code:

import numpy as np
from sklearn.linear_model import Perceptron

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([0, 1, 1, 0]) 

model = Perceptron (max_iter=1000, tol=1e-3)
model.fit(X, Y)

predictions = model.predict(X)
print("Input\tPrediction\tExpected\tRemark")
for i in range(len(X)):
    expected = Y[i]
    actual = predictions[i]
    remark = "Correct" if actual == expected else "May fail"
    print(f"{X[i]}\t{actual}\t\t{expected}\t\t{remark)")

    output:

    ![Deep learning test case 2](https://github.com/user-attachments/assets/1bc1a06b-e2f2-4ca6-ac51-4a6cf379657e)
