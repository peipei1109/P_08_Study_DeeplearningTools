#encoding: utf-8

'''
Created on 2017��2��17��

@author: Administrator
'''
import theano
import numpy
import theano.tensor as T

x=T.dmatrix('x')
s=1/(1+T.exp(-x))
logistic=theano.function([x],s)
print logistic([[0,1],[-1,-2]])
