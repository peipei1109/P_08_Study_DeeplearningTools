#encoding: utf-8

'''
Created on 2017��2��17��

@author: Administrator
'''
import theano
import theano.tensor as T
x=T.vector('x')
y=x**2
J,updates =theano.scan(lambda i,y,x: T.grad(y[i],x), sequences=T.arange(y.shape[0]),non_sequences=[y,x])  
f = theano.function([x], J, updates=updates) 

print f([4, 4])

g= theano.function([x],theano.gradient.jacobian(y,x))

print g([4, 4])


