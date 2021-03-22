#!/anacoda/envs/tensorflow/python
# -- coding = 'utf-8' --
# Python Version 3.7.9 # OS Windows 10
# @time : 2021/2/19 15:36
# copyright reserved

from sklearn import datasets
import matplotlib.pyplot as plt

faces = datasets.fetch_olivetti_faces()
print(faces.images.shape)

img_count = 0
plt.figure(figsize=(20, 20))
for img in faces.images:
    plt.subplot(20, 20, img_count+1)
    plt.imshow(img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(faces.target[img_count])
    img_count += 1
plt.show()