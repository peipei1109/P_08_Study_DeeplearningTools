#encoding: utf-8

'''
Created on 2017��2��17��

@author: Administrator
'''
import numpy as np
import theano.tensor as T
from theano import pp
from theano import  function
x=T.dscalar('x')
y=T.dscalar('y')
z=x+y

print(pp(z))

f=function([x,y],z)

print f(2,3)


