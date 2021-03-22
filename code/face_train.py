
#!/anacoda/envs/tensorflow/python
# -- coding = 'utf-8' --
# Python Version 3.7.9 # OS Windows 10
# @time : 2021/3/11 16:25
# copyright reserved

import numpy as np
from PIL import Image
import os
import cv2

# 人脸数据路径
path = '../Data/Face'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(
    r"C:/ProgramData/Anaconda3/envs/tensorflow/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f)
                  for f in os.listdir(path)]  # join函数的作用？
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert(
            'L')  # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split("_")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x: x + w])
            ids.append(id)
    return faceSamples, ids


print('Training faces. It will take a few seconds. Wait ...')
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write(r'../Data/trainer.yml')
print("{0} faces trained. Exiting Program".format(len(np.unique(ids))))