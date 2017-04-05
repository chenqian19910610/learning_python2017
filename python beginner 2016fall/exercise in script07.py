#Exercise block 1
"""
list=[]
for i in range(17):
    list.append(i**2)
print(list)

list=[]
for i in range(17):
    list.append(i**2)
sum=0
for x in list:
    sum=sum+x
print("sum of the list",sum)

list=[]
for i in range(17):
    list.append(i**2)
print(list)
new_list=[]
for x in list:
    if x>999 and x<10000:
        new_list.append(x)
print(new_list)

import csv
one_letter_code=[]
average_mass=[]
with open("files exercise/amino_acids.csv", "r", newline="") as fh:
    r=csv.reader(fh)
    header = next(r)
    for line in r:
        x=line[0]
        y=float(line[4])
        one_letter_code.append(x)
        average_mass.append(y)
    print(one_letter_code,average_mass)

import csv
one_letter_code=[]
average_mass=[]
with open("files exercise/amino_acids.csv", "r", newline="") as fh:
    r=csv.reader(fh)
    header = next(r)
    for line in r:
        x=line[0]
        y=float(line[4])
        one_letter_code.append(x)
        average_mass.append(y)
        m=input("pls input a random symbol")
        if m in one_letter_code:
            a=one_letter_code.index(m)
            print(m,"average mass is",average_mass[a])
        else:
            print("not valid symbol")

import csv
one_letter_code=[]
average_mass=[]
with open("files exercise/amino_acids.csv", "r", newline="") as fh:
    r=csv.reader(fh)
    header = next(r)
    for line in r:
        x=line[0]
        y=float(line[4])
        one_letter_code.append(x)
        average_mass.append(y)
m=input("pls input a random sequence")
sum=0
not_skip=0
for char in m:
    if char in one_letter_code:
        sum=sum+average_mass[one_letter_code.index(char)]
        not_skip=not_skip+1
print(sum, "skipped characters", len(m)-not_skip)

#Exercise block 3
import csv
one_letter_code=[]
match_code={}
with open("files exercise/amino_acids.csv", "r", newline="") as fh:
    r=csv.reader(fh)
    next(r)
    for line in r:
        match_code[line[0]]=float(line[4])
        one_letter_code.append(line[0])
m=input("pls enter a random symbol")
if m in one_letter_code:
    print(match_code[m])

import csv
one_letter_code=[]
match_code={}
with open("files exercise/amino_acids.csv", "r", newline="") as fh:
    r=csv.reader(fh)
    next(r)
    for line in r:
        match_code[line[0]]=float(line[4])
        one_letter_code.append(line[0])
m=input("pls enter a random sequence")
sum=0
not_skip=0
for char in m:
    if char in one_letter_code:
        sum=sum+match_code[char]
        not_skip=not_skip+1
print("weight of sequence is", sum, "unvalid character is", len(m)-not_skip)


tzt="this is some text, some insensible text indeed"
counts={}
for a in tzt.split(" "):
    if a not in counts.keys():
        counts[a]=1
    else:
        counts[a]=counts[a]+1
print(counts)
print(counts.keys())


#Exercise block 4
occurence={}
with open("files exercise/FASTA.txt","r",newline="") as fh:
    for line in fh:
        line=line.rstrip()
        if len(line)>0 and line[0]!=">":
            for symbol in line:
                if symbol not in occurence.keys():
                    occurence[symbol]=1
                else:
                    occurence[symbol]=occurence[symbol]+1
print(occurence)

container={}
with open("files exercise/Condo.txt","r",newline="") as fh_1:
    next(fh_1)
    for line in fh_1:
        fields = line.split(" ")
        if len(fields) > 1:
            container[fields[0]]=fields[1]
            container[fields[6]]=fields[7]
print(container)

groups=[1,0,0,1,1]
values=[1,2,3,2,7]
assignment={}
for i in range(len(values)):
    group=groups[i]
    value=values[i]
    if group not in assignment.keys():
        assignment[group]=[]
    assignment[group].append(value)
print(assignment)
"""

import csv
with open("files exercise/abc.csv","w",newline="") as fh:
    w=csv.writer(fh,delimiter=";")
    w.writerow(["group","value"])
