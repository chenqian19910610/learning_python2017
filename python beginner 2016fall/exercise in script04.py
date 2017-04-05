#exercise script 04
#exercise block 1
i=1
acc=0
for i in range(1,5):
	acc=acc+i*i
print("the sum of the first 4 square numbers", acc)

i=1
acc=1
for i in range(1,5):
	acc=acc*i
print("the product of the first 4 square numbers", acc)

i=1
acc=0
for i in range(0,11):
	acc=acc+2**i
print("the sum of 1+2+4+...+1024 is", acc)

i=1
for i in range(1,6):
	x="*"*i
	print(x)
print("the height of triangle is",i)

i=1
a=3
for i in range(1,5):
	x=" "*a+"*"*(7-2*a)+" "*a
	a=a-1
	print(x)
print("the height of triangle is",i)

#exercise block 2
acc=0
n=6
for i in range(n+1):
	if i%2==0:
	    acc=acc+i
print(acc)

acc_even=0
acc_odd=0
n=6
for i in range(n+1):
	if i%2==0:
	    acc_even=acc_even+i
	elif i%2!=0:
		acc_odd=acc_odd+i
print("the sum of even numbers", acc_even, "the sum of odd numbers",acc_odd)

n=6
for i in range(1,n+1):
	if i%2==0:
	    print(i,"is an even number")
	elif i%2!=0:
		print(i,"is an odd number")

n=30
for i in range(1,n+1):
	if i%3==0:
		print("fizz")
	elif i%5==0:
		print("buzz")
	else:
		print(i)

for i in range(1500,2001):
	if i%7==0 or i%13==0:
		print("the multiple of 7 or 13 is",i)

#optional task
found=False
for n in range(2,20):
	print(n)
	for i in range(2,n):
		if n%i==0:
			found=True
	print(not found)

#exercise block 3
txt="GCCGGA"
for i in range(len(txt)):
	if txt[i]=="G":
		print("found G at position","",i)

DNA="GGCTGGGAACTGGGAAGGCAGAGCCGCCGCCΑ"
acc=0
for i in range(len(DNA)):
	if DNA[i]=="A":
		acc=acc+1
	if DNA[i]=="G"and DNA[i+1]=="C":
		print("Found GC at position", i)
print("the total number of A", acc)

DNA="GGCTGGGAACTGGGAAGGCAGAGCCGCCGCCΑ"
n=len(DNA)
m=""
for i in range(len(DNA)):
	n=n-1
	m=m+DNA[n]
print(m)

#exercise block 4
txt="GCCGGA"
i=0
while i<len(txt):
	if txt[i]=="G":
		print("found G in position ",i)
	i=i+1

DNA="GGCTGGGAACTGGGAAGGCAGAGCCGCCGCCΑ"
i=0
m=""
n=len(DNA)
while i<len(DNA):
	n=n-1
	m=m+DNA[n]
	i=i+1
print(m)

#exercise block 5
sum=0
i=1 #the total input times
while True:
	x=float(input("pls input a random number"))
	sum=sum+x
	i=i+1
	if i>5:
		break
print("the sum of input numbers",sum, "the average",sum/i)

n=int(input("pls enter a random integer"))
i=0
while True:
	if n%2==0:
		n=n/2
		i=i+1
	else:
		n=3*n+1
		i=i+1
	if n==1:
		break
print("the total iteration time",i)

a=int(input("enter a"))
b=int(input("enter b"))
while a!=b:
	if a>b:
		a=a-b
	else:
		b=b-a
print("the common divisor",a)

i=5
sum=0
while i>0:
	x=float(input("enter a random number"))
	sum=sum+x
	i=i-1
print("the total",sum, "the average", sum/5)

i=5
sum=0
x=float(input("enter a random number"))
while i>0:
	y=float(input("enter a random number"))
	i=i-1
	if x>y:
		y=x
	else:
		x=y
print("the maximum value is",y)
n=int(input("pls enter a random integer"))
i=0
a=2
while n<100:
	if n%2==0:
		n=n/2
		i=i+1
	else:
		n=3*n+1
		i=i+1
	if n==1:
		break
if i>a:
	a=i
print("the maximum iteration time is,", a)

#exercise block 7
import random
a=random.randint(1,100)
x=int(input("pls enter the guessing number"))
i=0
print(a)
while x!=a:
	if a>x:
		print("the guessing is smaller")
	else:
		print("the guessing is bigger")
	x=int(input("pls enter the guessing number"))
	i=i+1
print("total guessing time",i)

#exercise block 7
while True:
	import random
	a=random.randint(1,100)
	x=int(input("pls enter the guessing number"))
	i=0
	print(a)
	while x!=a:
		if a>x:
			print("the guessing is smaller")
		else:
			print("the guessing is bigger")
		x=int(input("pls enter the guessing number"))
		i=i+1
	print("total guessing time",i)
	if i>10:
		break

#exercise block 7
import random
for i in range(20):
	a=random.randint(1,4)
	if a==1:
		a="A"
	elif a==2:
		a="G"
	elif a==3:
		a="T"
	else:
		a="C"
	print(a)

#exercise block 8
import random
x=input("player A")
y=random.randint(1,3)  #1="S", 2="P",3="R"
print(y)
if x=="R" and y==1:
	print("A wins")
elif x=="R" and y==2:
	print("B wins")
elif x=="R" and y==3:
	print("a tie round")
elif x=="S" and y==2:
	print("A wins")
elif x=="S" and y==3:
	print("B wins")
elif x=="S" and y==1:
	print("a tie round")
elif x=="P" and y==3:
	print("A wins")
elif x=="P" and y==1:
	print("B wins")
elif x=="P" and y==2:
	print("a tie round")
else:
	print("invalid round")
