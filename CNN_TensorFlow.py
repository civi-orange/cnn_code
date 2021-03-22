#!/anacoda/envs/tensorflow/python
# -- coding = 'utf-8' --
# Python Version 3.7.9 # OS Windows 10
# @time : 2021/2/22 11:18
# copyright reserved

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, \
    Flatten, Dense, BatchNormalization, Activation, Dropout
# 卷积核框架函数
# tf.keras.layers.Conv2D(filters= ,  # 卷积核个数
#                        kernel_size=,  # 卷积核大小
#                        strides=,  # 卷积步长
#                        padding=,  # 全零补充'same',默认不使用'valid'
#                        activation=,  # 'relu','sigmoid','tanh','softmax'  # 如有BN此处不写
#                        input_shape=  # 输入特征维度(高，宽，通道数) 可省略
#                        )
# model example
model_0 = tf.keras.models.Sequential([
    Conv2D(6, 5, padding='valid', activation='sigmoid'),
    MaxPool2D(2, 2),
    Conv2D(6, (5, 5), padding='valid', activation='sigmoid'),
    MaxPool2D(2, (2, 2)),
    Conv2D(filters=6, kernel_size=(5, 5),padding='valid', activation='sigmoid'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(10, activation='softmax')
])

# 批量标准化 Batch Normalization, BN
model_1 = tf.keras.models.Sequential([
    Conv2D(filters=6, kernel_size=(5, 5), padding='same'),  # 卷积层
    BatchNormalization(),  # BN层  卷积层之后激活层之前
    Activation('relu'),  # 激活层
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),  # 池化层
    Dropout(0.2),  # dropout层
])

# 最大值池化可提取图片纹理，均值池化可保留背景特征。
# tf.keras.layers.MaxPool2D(
#     pool_size=,  # 池化核尺寸，#正方形写核长整数，或（核高h，核宽w）
#     strides=,  # 池化步长，#步长整数， 或(纵向步长h，横向步长w)，默认为pool_size
#     padding=,  # ‘valid’or‘same’ #使用全零填充是“same”，不使用是“valid”（默认）
# )
# tf.keras.layers.AveragePooling2D(
#     pool_size=,  # 池化核尺寸，#正方形写核长整数，或（核高h，核宽w）
#     strides=,  # 池化步长，#步长整数， 或(纵向步长h，横向步长w)，默认为pool_size
#     padding=,  # ‘valid’or‘same’ #使用全零填充是“same”，不使用是“valid”（默认）
# )



