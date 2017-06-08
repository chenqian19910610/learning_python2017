# a,b,c,d=range(1,5)
# print(a,b,c,d)
#
# quotient, remainder=divmod(20,3)
# print(quotient,remainder)

# print("hello\nworld")   #换行
# print("hello\tworld")   #制表

# print(ord('A'))
# print(chr(66))

list_1=[1,1,12,[1,2,3]]
print(len(list_1))
print(list_1[-1])
list_1.append(13)
list_1.insert(0,53)
list_1.pop(1)
list_1[0]=66

for number in list_1:
    if number!="":
        continue     #提前结束，直接进入下一个循环
    print(number)