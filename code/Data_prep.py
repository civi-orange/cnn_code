
#!/anacoda/envs/tensorflow/python
# -- coding = 'utf-8' --
# Python Version 3.7.9 # OS Windows 10
# @time : 2021/2/19 17:20
# copyright reserved

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow.keras as keras

faces = datasets.fetch_olivetti_faces()
X = faces.images
y = faces.target
# print(X[0])
# print(Y[0])

X = X.reshape(400, 64, 64, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = keras.Sequential()
model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', input_shape=(64, 64, 1)))  # 第一层卷积层
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))  # 第二层卷积
model.add(keras.layers.Flatten())  # 压缩为一维数据
model.add(keras.layers.Dense(40, activation='softmax'))  # 全连接层

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

y_predict = model.predict(X_test)
print(y_test[0], np.argmax(y_predict[0]))