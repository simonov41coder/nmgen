import csv

companies = []

with open('Top 1000 technology companies.csv', newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        companies.append(row["Company"])

with open("companies.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(companies))

print("Saved", len(companies), "companies")
