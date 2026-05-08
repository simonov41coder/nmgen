import csv

companies = []

with open('sample.csv', newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        companies.append(row["title"])

with open("companies.txt", "a", encoding="utf-8") as f:
    f.write("\n".join(companies))

print("Saved", len(companies), "companies")
