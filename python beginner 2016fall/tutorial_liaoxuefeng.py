# a,b,c,d=range(1,5)
# print(a,b,c,d)
#
# quotient, remainder=divmod(20,3)
# print(quotient,remainder)

# print("hello\nworld")   #换行
# print("hello\tworld")   #制表

# print(ord('A'))
# print(chr(66))

# list_1=[1,1,12,[1,2,3]]
# print(len(list_1))
# print(list_1[-1])
# list_1.append(13)
# list_1.insert(0,53)
# list_1.pop(1)
# list_1[0]=66
#
# for number in list_1:
#     if number!="":
#         continue     #提前结束，直接进入下一个循环
#     print(number)

# table={'A':95,'B':100,'C':200}
# table['A']=25
# table.pop('C')
# print(table.get('D',-1)) #只能用key来索引，与value无关

# s1=set([1,2,3])
# s1.add(4)
# s1.remove(1)
# print(s1)
# s2=set([5,2,3,6])
# print(s1|s2,s1&s2)  #交集和并集


# def summup(n):
#     a=0
#     for i in range(n):
#         a=a+i**2
#     return a
# abc=int(input("enter a integer"))
# print(summup(abc))

"""
recursive function
"""
# def fact(n):
#     if n==1:
#         return 1
#     else:
#         return n*fact(n-1)
# print(fact(5))
#
# def hanoi(n,a,b,c):
#     if n==1:
#         print(a,"-->",c)
#     else:
#         hanoi(n-1,a,c,b)
#         print(a,"-->",c)
#         hanoi(n-1,b,a,c)
# hanoi(3,'A','B','C')

# a=list(range(100))
# print(a[1:25:3]) #截取一段切片数据

# from collections import Iterable  #判断是否可以迭代
# print(isinstance("dsaiydsafio",Iterable))
#
# for i,x in enumerate("dasuihdlkdhfsd"):   #同时迭代索引和值
#     print(i,x)

# list_3=[(1,2,3),(4,5,6),(7,8,9)]
# for x,y,z in list_3:
#     print(x,y,z)

# table_1={1:2,3:4,5:6,7:8,9:10}
# print([ x+y for x,y in table_1.items()])

#
# import os
# print([d for d in os.listdir('.')])  #打印出当前列表中的所有文件名

# list_1=[1]
# list_2=[1,1]
# n=0
# print(list_1)
# print(list_2)
# while n<=10:
#     list_3 = [1]
#     for i in range(len(list_2)-1):
#         list_3.append(int(list_2[i])+int(list_2[i+1]))
#     list_3.append(1)
#     print(list_3)
#     n=n+1
#     list_2=list_3        #杨辉三角形， list也相当于一个变量用，可以更新value


# L1=['AdAm','liSA','ChEn']
# def f1(L):
#     return L[0].upper()+L[1:].lower()
# L1=list(map(f1,L1))
# print(L1)
# from functools import reduce
# def f2(x,y):
#     return str(x)+str(y)
# L1=reduce(f2,L1)
# print(L1)

"""
partial function:新函数可以固定住原函数的部分参数
"""
# import functools
# int2=functools.partial(int,base=2)
# print(int2('1000000'))
#
# max2=functools.partial(max,10)
# print(max2(5,6,7))

import os
print(os.path.abspath(','))
print(__file__)
import sys
print(sys.argv)