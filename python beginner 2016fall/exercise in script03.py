"""
am_i_clever=True
print(type(am_i_clever))

print(3>4)
print(3==4)
print(not True or True)
"""
"""
print("abc"=="abc")
print("abc ">"abcd")
"""
"""
print(3 < 4 and 4 <= 5 and 5 != 6)
print(3 == 4 or 3 != 4 and 3 == 4)
print((3 == 4 or 3 != 4) and (3 == 4))
print("ab" <= "bcd" or 3 == 4)
print("ab" > "bcd" and 3 == 4)
print("" == " ")
print("abcd">" bcd")

# exercise block 1
if x>3:
    print("x is greater than 3", x)
print("x is", x)

x=int(input("pls enter a random number"))
if x>0:
    print("the number is positive")
else:
    print("the number is not positive")


x=float(input("pls enter a random number"))
if x>0.0 and x<1.0:
    print("the number is desired")
else:
    print("the number is not desired")


x=float(input("the total working hours:"))
y=float(input("the wage for each hour:"))
if x>40:
    z=(x-40)*y*2+40*y
else:
    z=40*x
print("the total wage is:",z)


# exercise block 2
x=int(input("pls enter a random integer:"))
if x>0:
    print(x,"is positive")
elif x<0:
    print(x,"is negative")
else:
    print(x,"is zero")

x=int(input("pls enter a random integer:"))
if x%3==0 and x%4==0:
    print(x,"is multiple of 12")
elif x%3==0:
    print(x,"is multiple of 3")
elif x%4==0:
    print(x, "is multiple of 4")
else:
    print(x,"is neither multiple of 3 or 4")

x=float(input("the total working hours:"))
y=float(input("the wage for each hour:"))
if x>60:
    z=(x-60)*y*3+40*y+20*y*2
elif x>40 and x<60:
    z=(x-40)*y*2+40*y
else:
    z=40(x-60)*y
print("the total wage is:",z)

x=float(input("the year"))
if x%4==0 and x%100!==0:
    print(x, "is a leap year")
else:
    print(x,"is a common year")

x = 1

if x > 0:
    print(x, "is positive")
elif x < 0:
    print(x, "is negative")
else:
    print(x, "is zero")

x = 1

if x > 0:
    print(x, "is positive")
if x < 0:
    print(x, "is negative")
else:
    print(x, "is zero")
"""

#Exercise block 3 (1)
x=input("player A")
y=input("player B")
if x=="R" and y=="S":
	print("A wins")
elif x=="R" and y=="P":
	print("B wins")
elif x=="R" and y=="R":
	print("a tie round")
elif x=="S" and y=="P":
	print("A wins")
elif x=="S" and y=="R":
	print("B wins")
elif x=="S" and y=="S":
	print("a tie round")
elif x=="P" and y=="R":
	print("A wins")
elif x=="P" and y=="S":
	print("B wins")
elif x=="P" and y=="P":
	print("a tie round")
else:
	print("invalid round")

#Exercise block 3 (2)
x=input("player A")
y=input("player B")
if x!="R" and x!="S" and x!="P" or y!="R" and y!="S" and y!="P":
	print("invalid runs")
elif x=="R" and y=="P":
	print("B wins")
elif x=="R" and B=="R":
	print("a tie round")
elif x=="S" and y=="P":
	print("A wins")
elif x=="S" and y=="R":
	print("B wins")
elif x=="S" and y=="S":
	print("a tie round")
elif x=="P" and y=="R":
	print("A wins")
elif x=="P" and y=="S":
	print("B wins")
elif x=="P" and y=="P":
	print("a tie round")














