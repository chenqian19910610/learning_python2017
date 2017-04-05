#Block 1
"""
def my_name():
    a="Chen"
    b="Qian"
    print("my name is",a,b)
my_name()

#Block 2
def greet(a,b):
    print("hello",a,b)
greet("Chen","Qian")

#Block 3
def average_of_three(a,b,c):
    m=(a+b+c)/3
    print(m)
    return m
average=average_of_three(1,2,3)
print(average)

#Block 4
def collatz(n):
    sum_of_iteration=0
    while n!=1:
        if n%2==0:
            n=n/2
        else:
            n=3*n+1
        sum_of_iteration=sum_of_iteration+1
    print(sum_of_iteration)
    return sum_of_iteration
x=collatz(17)
print(x)


#Block 5
def maximum(list):
    a=0
    for i in range(len(list)):
        if a<int(list[i]):
            a=int(list[i])
    print(a)
    return a
x=maximum([1,2,3])
print(x)

def standdev(list):
    sum_of_number = 0
    total_number = 0
    sum_of_dev = 0
    for i in range(len(list)):
        sum_of_number=sum_of_number+int(list[i])
        total_number=total_number+1
    mean=sum_of_number/total_number
    for i in range(len(list)):
        sum_of_dev=sum_of_dev+(int(list[i])-mean)**2
    result=(sum_of_dev/(total_number-1))**0.5
    return result
x=standdev([3,4,5,6,7])
print(x)


def max(list):
    if list==[]:
        return None
    else:
        max_sofar = list[0]
        for i in range(1, len(list)):
            value= list[i]
            if value> max_sofar:
                max_sofar=value
    return max_sofar

x=max([1,2,3])  # function should be called outside def...
y=max([])
print(x,y)
"""

import pylab
xvalues = []
yvalues = []
for x in range(-5 , 5):
    y = 6 * x + 5 - 6 * x * x
    xvalues.append(x)
    yvalues.append(y)

pylab.plot(xvalues, yvalues, "r")  #red lines
pylab.plot(xvalues, yvalues, "bo")   # blue circles, g* green star
pylab.show()




























