# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 01:27:24 2016

@author: ceciliaLee
"""

import theano.tensor as T
import theano
"""
 function(inputs, outputs, mode=None, updates=None, givens=None, no_default_updates=False, 
 accept_inplace=False, name=None,rebuild_strict=True, allow_input_downcast=None, 
 profile=None, on_unused_input='raise')
 """
 
x=T.iscalar('x') #声明一个int类型的变量x  
y=T.pow(x, 3) #定义y=x^3  
f1=theano.function([x], y) #定义函数的自变量为x（输入），因变量为y（输出）  
print f1(2) ## Call function
print f1(3) ## Call function


## Sigmoid function
a =T.fscalar('a')  #定义一个float类型的变量a  
b = 1 / (1 + T.exp(-a))  #定义变量b
f2 = theano.function([a],b)    #定义函数f，输入为a，输出为b  
print f2(3)       # Call  

# partial derivative
c =T.fscalar('c')    #定义一个float类型的变量c
d = 1 / (1 + T.exp(-c))   #定义变量d  
dx=theano.grad(d,c)    #偏导数函数  
f3 = theano.function([c],dx)#定义函数f，输入为x，输出为s函数的偏导数  
print f3(3)   #计算当x=3的时候，函数y的偏导数

## Shared values
# 在程序中，我们一般把神经网络的参数W、b等定义为共享变量，因为网络的参数，
# 基本上是每个线程都需要访问
import numpy as np
mtr=np.random.randn(3,4)  # generate a matrix randomly
sv=theano.shared(mtr) ## create shared value sv from mtr
print sv.get_value()  ## 通过get_value()、set_value()可以查看、设置共享变量的数值


## Update shared values
# theano.function函数，有个非常重要的参数updates，updates是一个包含
# 两个元素的列表或tuple，updates=[old_w,new_w]，当函数被调用的时候，
# 这个会用new_w替换old_w

w = theano.shared(1)  # define a shared value w, and the initialized value 1
u =T.iscalar('v')  
f4 =theano.function([u], w, updates=[[w, w+u]])#定义函数自变量为u，因变量为w，当函数执行完毕后，更新参数w=w+u 
print f4(3) #函数输出为w, u=3
print w.get_value()#这个时候可以看到w=w+x为4  


