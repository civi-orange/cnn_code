
#!/anacoda/envs/tensorflow/python
# -- coding = 'utf-8' --
# Python Version 3.7.9 # OS Windows 10
# @time : 2021/3/15 14:43
# copyright reserved

import tensorflow as tf
import os
import Model_C_R

check_path = '../path/model.ckpt'
check_dir = os.path.dirname(check_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True, verbose=1)
model = Model_C_R.create_model()
model.fit(Model_C_R.train_images, Model_C_R.train_labels, epochs=10,
          validation_data=(Model_C_R.test_images, Model_C_R.test_labels),
          callbacks=[cp_callback])

loss, acc = model.evaluate(Model_C_R.test_images, Model_C_R.test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

model.load_weights(check_path)
loss_, acc_ = model.evaluate(Model_C_R.test_images, Model_C_R.test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc_))
