"""test..........
a=input("pls input a random string")
b=a.upper()
for i in range(len(b)):
    if b[i]==a[i]:
        print("it is upper case")
    else:
        print("need upper case")


a=input("pls input a nucleotide sequence")
b=len(a)
if a!=a.upper():
    print("the sequence is not valid")
else:
    m=a.count("GC")/b
    print("the percentage of 'GC' is", m)


a="abc"
if (' ' in a) == True:
    print("Hello")
else:
    print("NO Hello")
m=a.count(" ")
print(m)
"""

# exercise BLOCK 2.2
"""
a=input("a random sequence")
for i in range(len(a)):
	if a[i]!=a[i].upper():
		print("the",i,"th position is not uppercase")
	else:
		print("the",i,"th position is uppercase")


#BLOCK COMBINING 2.3,2.4,2.5,2.6
a=input("a nucleotide sequence")
a=a.upper()
a=a.replace(" ","")
while True:
	if a.count("T")+a.count("G")+a.count("A")+a.count("C")!=len(a):
		a=input("a nucleotide sequence")
	else:
		b=(a.count("GC"))/len(a)*2*100
		print("the sequence is appropriate",a,"the percentage of GC is",b)
		break


#BLOCK 3.2
a=input("a nucleotide sequence")
is_palindrome=True
for i in range(len(a)):
	if a[i]!=a[len(a)-1-i]:
		print("is not a palindrome")
		is_palindrome=False
		break
if is_palindrome:
	print("is a palindrome")

#BLOCK 3.4
a=input("a random sequence")
sum=0
for i in range(len(a)):
	if a[i]==" ":
		sum=sum+1
print("the total space is",sum)


#BLOCK 3.5
a=input("a nucleotide sequence")
a=a.upper()
a=a.replace(" ","")
for i in range(len(a)):
	if a[i]=="G" and a[i+1]=="C":
		print("found GC starting at position",i+1)

#BLOCK 3.6
a=input("a nucleotide sequence")
a=a.upper()
a=a.replace(" ","")
while True:
	if a.count("T")+a.count("G")+a.count("A")+a.count("C")!=len(a):
		a=input("a nucleotide sequence")
	else:
		a=list(a)
		for i in range(len(a)):
			if a[i]=="T":
				a[i]="A"
			elif a[i]=="A":
				a[i]="T"
			elif a[i]=="G":
				a[i]="C"
			else:
				a[i]="G"
		b=''.join(a)
		print("complement sequence",b)
		break
c=list(b)
for i in range(len(a)):
	print("the", i+1, "position of reverse complement is", c[len(c)-i-1])


#number guessing game
import random
for i in range(1):
	a=random.randint(1,100)
print(a)
m=int(input("pls enter a random number"))
while True:
	if a>m:
		print("failed")
		ask=input("want try again")
		if ask=="NO":
			break
		else:
			m=int(input("pls enter a larger number"))
	elif a<m:
		print("failed")
		ask=input("want try again")
		if ask=="NO":
			break
		else:
			m=int(input("pls enter a larger number"))
	else:
		print("successful guessing")
		break

#Block 4.1,4.2
a=input("a random string")
new_a=""
for i in range(len(a)):
	b=a[i]
	code=ord(b)
	if code>=97 and code<=122:
		code=code-32
	new_a=new_a+chr(code)
print("the uppercase string", new_a)

a=input("a random string")
new_a=""
for i in range(len(a)):
	b=a[i]
	code=ord(b)
	if code>=35 and code<=90:
		code=code+32
	new_a=new_a+chr(code)
print("the lowercase string", new_a)


#Block 4.3,4.4
a=input("a random string")
a=a.upper()
new_a=""
for i in range(len(a)):
	b=a[i]
	code=ord(b)
	if code>=65 and code<90:
		code=code+1
	else:
		code=65
	new_a=new_a+chr(code)
print("the rot-1 cipher", new_a)
inverse_name=""
for i in range(len(new_a)):
	m=new_a[len(new_a)-i-1]
	inverse_name=inverse_name+m
print("the inverse name is", inverse_name)


#Block 4.5
a=input("a random string")
a=a.upper()
new_a=""
for i in range(len(a)):
	b=a[i]
	code=ord(b)
	if code>=65 and code<=87:
		code=code+1
	elif code==88:
		code=65
	elif code==89:
		code=66
	else:
		code=67
	new_a=new_a+chr(code)
print("the caesar cipher is",new_a)

#Block 5.1,5.2
a=input("pls input a random sequence")
a=a.upper()
a=a.replace(" ","")
b="GC"
position=a.find(b)
while position>-1:
	print("the position of GC is",position)
	position=a.find(b,position+1)

#Block 5.3
a=input("pls input a random sequence")
a=a.upper()
a=a.replace(" ","")
b="GC"
position=a.find(b,0)
while True:
	if position<-1:
		print("not found")
		break
	else:
		i=position
		while i<len(a) and position>-1:
			print("the position of GC is",position)
			i=i+1
			position=a.find(b,position+1)
"""
