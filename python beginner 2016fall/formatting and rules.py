"""
format in python
"""
a='{1} {2} {0}'.format('one','two','three')
print(a)

a='{:>10}'.format('test') # align to the right
a='{:^10}'.format('test') # align to the middle
print(a)

a='{:.3}'.format('abcdefghijk') # truncating words to specified length
a='{:>10.3}'.format('abcdefghijk') # combine truncating and padding
print(a)

numbers=[23.00,0.123545684,1,4.2,9887.3] #floats
for number in numbers:
    print('{:10.4f}'.format(number))

a=round(12358546546, -5) # round integers
print(a)

a='{:+d}'.format(42) # print signs
b='{:d}'.format(-42)
print(a,b)

data={'first':'abc','second':'123'} #named placeholders
print('{first}{second}'.format(**data))

person={'firstname':'Qian','lastname':'Chen'} # how to get information !!!
print('{p[firstname]} {p[lastname]}'.format(p=person))
data=[4,5,6,7,8]
print('{d[0]} {d[3]}'.format(d=data))

class Plant(object):
    type='tree'
    kinds=[{'name':'oak'},{'name':'maple'}]
a='{p.type}:{p.kinds[0][name]}'.format(p=Plant()) #按照给定的格式输出指定的值
print(a)

from datetime import datetime #print time
a='{:%Y-%m-%d %H:%M}'.format(datetime(2017,4,30,16,25))
print(a)

a='{:{prec}}={:{prec}}'.format('chenqian',2.7184568,prec='.3')#truncate with parameterized format
print(a)

"""
numpy in python
"""
import numpy as np
a=np.array([1,2,3])
b=np.transpose(a)
print(np.dot(a,b))

import sympy
from sympy import *
x=sympy.Symbol('x')
y=sympy.Symbol('y')
m=sympy.simplify(x+y+x-y)
a=sympy.limit(sympy.sin(x),x,0)
b=sympy.diff(sympy.tanh(x),x)
print(m,a,b)

def f(x):
    y=sympy.sin(x)*sympy.tanh(x)
    return y
a=sympy.diff(f(x),x)
b=sympy.lambdify(x,a)  # transform to lambda function to make calculation faster
c=b(6)
print(a,'the derivative is',c)

a=sympy.integrate(3*x**5+sympy.sin(x),x)
print(a)

M=sympy.zeros(3,5)
I=sympy.eye(3)
print(M,I)

import numpy
from numpy import array
from numpy.linalg import inv,det
M=[[25,67],[13,51]]
N=array(M)
print(det(M),inv(M),N)

