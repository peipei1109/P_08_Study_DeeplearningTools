#encoding: utf-8

'''
Created on 2017��2��17��

@author: Administrator
'''
import theano
import theano.tensor as T
from theano import function
from theano import shared

state =shared(0)

inc =T.iscalar('inc')
accumulator=function([inc],state, updates=[(state,state+inc)])
print accumulator(10)
print state.get_value()

new_state = shared(0)

new_accumulator = accumulator.copy(swap={state:new_state})
print new_accumulator(100)
print(new_state.get_value())
print(state.get_value())

# null_accumulator = accumulator.copy(delete_updates=True) existing bug, unused input error

# print null_accumulator(90) 

print state.get_value()
