# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 01:41:43 2016

@author: ceciliaLee
"""

import numpy as np 
import theano  
import theano.tensor as T  
# rng = numpy.random  
  
N = 10     # generate 10 samples, each is 3-dimension vector. Used to train
dim = 3  
D = (np.random.randn(N, dim).astype(np.float32), np.random.randint(size=N, low=0, high=2).astype(np.float32))  
  
  
# Traning data、and the corresponding labeks (For training)  
data = T.matrix("data")  
label = T.vector("label")  
  
# Shared values w and b,initialize w randomly and set b as 0
w = theano.shared(np.random.randn(dim), name="w")  
b = theano.shared(0., name="b")  
  
# Construct loss function
sig = 1 / (1 + T.exp(-T.dot(data, w) - b))   # Sigmoid Function  
xEn = -label * T.log(sig) - (1-label) * T.log(1-sig) # Cross-entropy loss function
cost = xEn.mean() + 0.01 * (w ** 2).sum()# 损失函数的平均值+L2正则项，其中权重衰减系数为0.01  
gw, gb = T.grad(cost, [w, b])             #对总损失函数求参数的偏导数  
  
prediction = sig > 0.5                    # predict
  
train = theano.function(inputs=[data,label],outputs=[prediction, xEn],updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))#训练所需函数  
predict = theano.function(inputs=[data], outputs=prediction)  #测试阶段函数  
  
# Training
training_steps = 1000  
for i in range(training_steps):  
    pred, err = train(D[0], D[1])  
    print err.mean()  #查看损失函数下降变化过程  