import csv

with open('adult.csv', 'r', newline='') as infile:
    reader = csv.reader(infile)
    
    with open('census.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        for index, row in enumerate(reader, start=1):
            writer.writerow([index] + row)