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
import numpy
A=[[1,2],[2,3]]
B=[[3,4],[4,5]]
print(numpy)