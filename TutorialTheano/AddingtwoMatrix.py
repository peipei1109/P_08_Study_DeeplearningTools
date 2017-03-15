#encoding: utf-8

'''
Created on 2017��2��17��

@author: Administrator
'''
import numpy as np
import theano.tensor as T
from theano import pp
from theano import  function
x=T.dmatrix('x')
y=T.dmatrix('y')
z=x+y

print(pp(z))

f=function([x,y],z)

print f([[1, 2], [3, 4]], [[10, 20], [30, 40]])


a=T.dvector()
out=a+a**10
print pp(out)
f=function([a],out)
print f([0,1,2])


