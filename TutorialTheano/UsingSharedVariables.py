#encoding: utf-8

'''
Created on 2017��2��17��

@author: Administrator
'''

import theano
import numpy
from theano import shared
from theano import function
import theano.tensor as T


state =shared(0)
inc =T.iscalar('inc')
accumulator=function([inc],state, updates=[(state,state+inc)])

print state.get_value()

print accumulator(1)

print state.get_value()

accumulator(300)

print state.get_value()


state.set_value(-1)

accumulator(3)

print state.get_value()


decrementor = function([inc],state,updates=[(state, state-inc)])

print decrementor(2)

print state.get_value()

fn_of_state = state * 2 + inc                

# The type of foo must match the shared variable we are replacing
#with the 'givens'

foo=T.scalar(dtype=state.dtype)
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])

print skip_shared(1, 3) #we are using 3 for the state, not state.value

print state.get_value() ## old state still there, but we didn't use it
