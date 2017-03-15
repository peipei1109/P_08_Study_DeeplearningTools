#encoding: utf-8

'''
Created on 2017��2��17��

@author: Administrator
'''
import theano
import theano.tensor as T

x=T.dvector('x')
y=x**2
cost=y.sum()
gy=T.grad(cost,x)
H, updates = theano.scan(lambda i,gy,x: T.grad(gy[i],x), sequences=T.arange(gy.shape[0]), non_sequences=[gy,x])
f=theano.function([x],H,updates=updates)
print f([4, 4])

g=theano.function([x], T.hessian(cost, x))
print g([4, 4])
