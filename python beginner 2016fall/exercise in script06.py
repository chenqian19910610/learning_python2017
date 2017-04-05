"""
import csv

with open("files exercise/amino_acids.csv", "r", newline="") as fh_in:
    r = csv.reader(fh_in)
    with open("files exercise/amino_acids2.csv", "w", newline="") as fh_out:
        w = csv.writer(fh_out)
        for row in r:
            row.append("NEW COLUMN")
            w.writerow(row)

import csv
with open("files exercise/amino_acids.csv","r",newline="") as fh_out:
    with open("files exercise/amino_acids_1.csv","w",newline="") as fh_in:
        r=csv.reader(fh_out)
        w=csv.writer(fh_in,delimiter=";")   # use delimiter; to separate columns ~
        for line in r:
        a=line[3]
        sum_mono=sum_mono+float(a)
        row_number=row_number+1
    print("average monoisotpic is", sum_mono/row_number)

import csv
with open("files exercise/amino_acids.csv","r",newline="") as fh_in:
    r=csv.reader(fh_in)
    header=next(r)
    print(header)
    mass1 = 0
    cells = list(r)
    print(cells[1][2])
"""

import csv
with open("files exercise/amino_acids.csv","r",newline="") as fh_out:
    with open("files exercise/amino_acids_1.csv","w",newline="") as fh_in:
        r=csv.reader(fh_out)
        w=csv.writer(fh_in,delimiter=";")   # use delimiter; to separate columns ~
        header=next(r)   # print header and append the header before the loop
        header.append("x")
        header.append("y")
        w.writerow(header)

        for line in r:
            x = float(line[3])
            line.append(x * 2)
            y=float(line[4])
            line.append(y + 1)
            w.writerow(line) # print in loop should be at last





