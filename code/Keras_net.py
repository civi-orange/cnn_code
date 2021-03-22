@@ -0,0 +1,37 @@
#!/anacoda/envs/tensorflow/python
# -- coding = 'utf-8' --
# Python Version 3.7.9 # OS Windows 10
# @time : 2021/3/12 14:16
# copyright reserved

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

datapath = 'C:/Person/GitHub/cnn_code/Data/mnist/mnist.npz'
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(datapath)

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

i = 88
plt.imshow(x_test[i], cmap=plt.cm.binary)
plt.show()
predictions = model.predict(x_test)
print(np.argmax(predictions[i]))