
#block 1
hours=float(input("number of hours worked:"))
salary_per_hour=float(input("salary per hour"))
if hours<=40:
    salary=hours*salary_per_hour
else:
    salary_40=40*salary_per_hour
    over_hours=hours-40
    over_hours_salary=2*over_hours*salary_per_hour
    salary=salary_40+over_hours_salary
print("the calculated salary is:", salary)


#block 2
print(2**3)


print(2.21e17/1.1e-3)

print("overall weight of rice in ton", (2**64-1)*(25e-3)/(10**6),"percentage in relation to earth weight",(2**64-1)*(25e-3)/(5.972*10**21))

import math
print(math.sin(2),math.pi,math.e, math.cos(math.pi),math.log(math.e))

#block 3
import math
a=(math.pi)**(math.e)
b=(math.e)**(math.pi)
if a>=b:
    print("true")
else:
    print("false")

a=1
b=2
c=math.sqrt(a**2+b**2)
print("the hypotenuse is", c)

x=5.2
x0=-4.8
k=2.23
L=32.42
print("the logistic curve f(x) is", L/(1+math.e**(-k*(x-x0))))


#block 4
print(len("abcdedf"))
print("123"+"456",123+456)

#block 5
x=float(input("pls enter the temperature in Celsius:"))
print ("temperature in Farenheit is:", x*1.8+32)

