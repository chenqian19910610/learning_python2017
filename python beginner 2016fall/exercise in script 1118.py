"""
#Block 2
fh=open("IWCCE 2017.docx","rb")
content=fh.read()
print(repr(content))
fh.close()

fh=open("numbers.txt","w")
for char in "abcdes":
    print(char,file=fh)
fh=open("numbers.txt","r")
content=fh.read()
print(repr(content))
fh.close()

fh=open("numbers.txt","r")
for line in fh:
    print(line)
fh.close()
fh=open("numbers.txt","r")
for line in fh:
    print(repr(line))
fh.close()

#Block 3
fh=open("numbers.txt","w")
for i in range(1,11):
    print(i, file=fh)
fh.close()
sum=0
fh=open("numbers.txt","r")
for line in fh:
    sum=sum+int(line)
print("the total of lines",sum)
fh.close()

fh=open("file handle.txt","r")
for line in fh:
    if line[0]==">":
        print(line.rstrip())
fh.close()

#Block 4
with open("file handle.txt","r") as fh:
    for line in fh:
        if line[0]==">":
            print(line.rstrip())

with open("file handle.txt","r") as fh:
    with open("description lines.txt","w") as fh_1:
        for line in fh:
            if line[0]==">":
                print(line.rstrip(), file=fh_1)

#Block 5
count=0
with open("file handle.txt","r") as fh:
    for line in fh:
        line=line.rstrip()
        if  len(line)>0 and line[0]==">": #why len(line) should go first??
            last_status=line
            count=0
        elif line=="":
            print(count,last_status)
        else:
            count=count+len(line)

with open("file handle.txt","r") as fh:
    with open("description lines.csv","w") as csv:
        for line in fh:
            line=line.rstrip()
            if len(line)>0 and line[0]==">":
                last_status=line
                count=0
            elif line=="":
                print(count,last_status,file=csv) # but how to arrange the prints in two coloumns??
            else:
                count=count+len(line)

with open("file handle.txt","r") as fh:
    with open("description lines.csv","w") as csv:
        for line in fh:
            line=line.rstrip()
            if len(line)>0 and line[0]==">":
                last_status=line
                count=0
            elif line=="":
                print(count,last_status,file=csv) # but how to arrange the prints in two coloumns??
            else:
                count=count+len(line)
                percentage = line.count("GC") / len(line)
                print("GC content is", percentage,file=csv)

#Block 5 optional
with open("file handle.txt","r") as fh:
    with open("description lines.csv","w") as csv:
        for line in fh:
            if line[0]==">":
                count=0
                last_status=line
            else:
                count=count+len(line)
                percentage = line.count("GC") / len(line)
                print("GC content is", last_status, percentage,count,file=csv)

with open("C:\\Users\\Administrator\\Desktop\\abc.txt","w") as abc:
    for i in range(0,11):
        print(i, file=abc)

#Block 6
import csv
with open("example.csv","w",newline="")as fh:
    w=csv.writer(fh)
    w.writerow(["a","b","c"])
    w.writerow([1,2,","])
    w.writerow([2,3,7])
with open("example.csv","r",newline="")as fh:
    for line in fh:
        print(line)

import csv
line_number=0
with open("example.csv","r",newline="") as fh:
        r=csv.reader(fh)
        header=next(r)
        print("header is",header)
        for line in r:
            a=line[0]
            b=line[1]
            print("a+b",int(a)+int(b))

import csv
with open("description lines.csv", "r",newline="") as fh:
    for line in csv.reader(fh):
        if len(line) > 0 and line[0] == ">":
            last_status = line
            count = 0
        elif line == "":
            print(count, last_status)  # but how to arrange the prints in two coloumns??
        else:
            count = count + len(line)
            percentage = line.count("GC") / len(line)
            print("GC content is", percentage)

import csv
with open("amino_acids.csv","r",newline="") as fh:
    for line in csv.reader(fh):
        print(line)

import csv
with open("amino_acids.csv","r",newline="") as fh:
    r=csv.reader(fh)
    header=next(r)
    count=0
    sum=0
    for line in r:
        a = line[3]
        sum=sum+float(a)
        count=count+1
    print("average Monoisotopic is",sum/count)

m=input("pls input a one-letter code of an amino acid")
import csv
with open("amino_acids.csv","r",newline="") as fh:
    r=csv.reader(fh)
    header=next(r)
    result="false"
    for line in r:
        name=line[0]
        chemical=line[2]
        if m==name:
            print(name,chemical)
            result = "Yes"
    if result!="Yes":
        print("input amino acid not found")

import csv
with open("files exercise/amino_acids.csv","r",newline="") as fh:
    r=csv.reader(fh)
    header=next(r)
    a=0
    for line in r:
        b=float(line[3])
        if a<b:
            a=b
    print("the max is", b)
"""