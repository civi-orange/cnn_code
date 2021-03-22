#!/anacoda/envs/tensorflow/python
# -- coding = 'utf-8' --
# Python Version 3.7.9 # OS Windows 10
# @time : 2021/2/20 9:53
# copyright reserved

import numpy as np
import h5py
import matplotlib.pyplot as plt

# # import tensorflow as tf  # 制作图片路径CSV数据
# import os
#
# path = "Image"
# filename = os.listdir(path)
# str_header = "path,class"
# str_text = ""
#
# with open('train_list.csv', 'w') as fid:
#     fid.write(str_header + '\n')
#     for index in range(len(filename)):
#         str_text = path + os.sep + filename[index] + ',' + '1' + '\n'  # os.sep = \
#         fid.write(str_text)
#     fid.close()


plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)


# 填充pad
# padding操作需要使用到numpy中的一个函数：np.pad()。假设我们要填充一个数组a，维度为（5,5,5,5,5），
# 如果我们想要第二个维度的pad=1，第4个维度的pad=3，其余pad=0，那么我们如下使用np.pad()
# a = np.pad(a, ((0,0),(1,1),(0,0),(3,3),(0,0)), 'constant', constant_values = (...,...))
def zero_pad(X, pad):  # 矩阵周边填充
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    return X_pad


# # 函数操作示意
# x = np.random.randn(4, 3, 3, 2)
# x_pad = zero_pad(x, 2)
# print('x.shape=', x.shape)
# print('x_pad.shape=', x_pad.shape)
# print('x[1,1]=', x[1, 1])
# print('x_pad[1,1]=', x_pad[1, 1])
#
# fig, axarr = plt.subplots(1,2)
# axarr[0].set_title('x')
# axarr[0].imshow(x[0, :, :, 0])
# axarr[1].set_title('x_pad')
# axarr[1].imshow(x_pad[0, :, :, 0])
# plt.show()

# 单步卷积，==卷积计算方法
def conv_single_step(a_slice_prev, W, b):
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + b
    return Z


# # test 卷积计算  卷积维度：（原维度-卷积核维度+2*补边）/步长+1
# a_slice_prev = np.random.randn(4, 4, 3)
# W = np.random.randn(4, 4, 3)
# b = np.random.randn(1, 1, 1)
# Z = conv_single_step(a_slice_prev, W, b)
# print(a_slice_prev)
# print(W)
# print(b)
# print(Z)

# 如果想从矩阵a_prev（形状为（5,5,3））中截取一个（2,2）的片，我们可以
# a_slice_prev = a_prev[0:2,0:2,:]
# a_slice_prev的四角vert_start, vert_end, horiz_start 和 horiz_end

# 卷积层 函数代码
def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape  # 目标
    (f, f, n_C_prev, n_C) = W.shape  # 卷积核

    step = hparameters['step']  # 卷积步长
    pad = hparameters['pad']  # 卷积补0

    n_H = int((n_H_prev - f + 2 * pad) / step + 1)  # 维度计算
    n_W = int((n_W_prev - f + 2 * pad) / step + 1)  # 维度计算

    Z = np.zeros((m, n_H, n_W, n_C))  # 结果矩阵
    A_prev_pad = zero_pad(A_prev, pad)  # 周边填充pad宽

    for i in range(m):  # 遍历图像数
        a_prev_pad = A_prev_pad[i, :, :, :]
        for h in range(n_H):  # 遍历高度
            for w in range(n_W):  # 遍历宽度
                for c in range(n_C):  # 遍历深度
                    vert_start = step * h
                    vert_end = vert_start + f
                    horiz_start = step * w
                    horiz_end = horiz_start + f
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

    assert (Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)
    return Z, cache


# 测试卷积层代码数据
A_prev_ = np.random.randn(10, 4, 4, 3)
W_ = np.random.randn(2, 2, 3, 8)
b_ = np.random.randn(1, 1, 1, 8)
hparameters_ = {"pad": 2, "step": 2}

Z_, cache_conv = conv_forward(A_prev_, W_, b_, hparameters_)
print("Z's mean=", np.mean(Z_))
print("Z[3,2,1]=", Z_[3, 2, 1])
print("cache_conv[0][1][2][3]=", cache_conv[0][1][2][3])


# 池化层
def pool_forward(A_prev, hparameters, model='max'):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hparameters['f']
    step = hparameters['step']

    n_H = int(1 + (n_H_prev - f) / step)
    n_W = int(1 + (n_W_prev - f) / step)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = h * step
                    vert_end = vert_start + f
                    horiz_start = w * step
                    horiz_end = horiz_start + f

                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    if model == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif model == 'average':
                        A[i, h, w, c] = np.mean(a_prev_slice)
    assert (A.shape == (m, n_H, n_W, n_C))
    cahce = (A_prev, hparameters)
    return A, cahce


# # 池化测试
# A_prev_ = np.random.randn(2, 4, 4, 3)
# hparameters_ = {"step": 2, "f": 3}
#
# A, cache = pool_forward(A_prev_, hparameters_)
# print("mode = max")
# print("A=", A)
# print()
#
# A1, cache1 = pool_forward(A_prev_, hparameters_, model="average")
# print("mode = average")
# print("A=", A1)


# 卷积层反向传播过程
def conv_backward(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    step = hparameters['step']
    pad = hparameters['pad']
    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):

        a_prev_pad = A_prev_pad[i, :, :, :]
        da_prev_pad = dA_prev_pad[i, :, :, :]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * step
                    vert_end = vert_start + f
                    horiz_start = w * step
                    horiz_end = horiz_start + f
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    return dA_prev, dW, db


# A_prev_ = np.random.randn(10, 4, 4, 3)
# W_ = np.random.randn(2, 2, 3, 8)
# b_ = np.random.randn(1, 1, 1, 8)
# hparameters_ = {"pad": 2, "step": 2}
#
# Z_, cache_conv = conv_forward(A_prev_, W_, b_, hparameters_)
# dA_, dW_, db_ = conv_backward(Z_, cache_conv)
# print("dA_mean =", np.mean(dA_))
# print("dW_mean =", np.mean(dW_))
# print("db_mean =", np.mean(db_))


# 池化层反向传播
# 定位max池化，最大值元素位置
def create_mask_from_window(x):
    mask = (x == np.max(x))
    return mask


# # A[i, j] = True if X[i, j] == x
# X = np.random.randn(2, 3)
# mask_ = create_mask_from_window(X)
# print(X)
# print(mask_)

# 均值池化，各元素权重定位
def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = dz/(n_H*n_W)
    a = average*np.ones(shape)
    return a


def pool_backward(dA, cache, mode='max'):
    (A_prev, hparameters) = cache
    step = hparameters['step']
    f = hparameters['f']
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    dA_prev = np.zeros(m, n_H_prev, n_W_prev, n_C_prev)
    for i in range(m):
        a_prev = A_prev[i, :, :, :]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = step * h
                    vert_end = vert_start + f
                    horiz_start = step * w
                    horiz_end = horiz_start + f
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                    elif mode == "average":
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
    assert (dA_prev.shape == A_prev.shape)
    return dA_prev


# A_prev_ = np.random.randn(5, 5, 3, 2)
# hparameters_ = {"step": 1, "f": 2}
# A_, cache_ = pool_forward(A_prev_, hparameters_)
# dA_ = np.random.randn(5, 4, 2, 2)
#
# dA_prev_m = pool_backward(dA_, cache_, mode="max")
# print("mode = max")
# print('mean of dA = ', np.mean(dA_))
# print('dA_prev[1,1] = ', dA_prev_m[1, 1])
# print()
# dA_prev_a = pool_backward(dA_, cache_, mode="average")
# print("mode = average")
# print('mean of dA = ', np.mean(dA_))
# print('dA_prev[1,1] = ', dA_prev_a[1, 1])




