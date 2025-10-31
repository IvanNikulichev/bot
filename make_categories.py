import csv

input_file = "Pio.txt"  
output_file = "catigories.csv"  

with open(input_file, "r", encoding="utf-8") as f_in:
    lines = [line.strip() for line in f_in if line.strip()]

# Пишем в CSV
with open(output_file, "w", encoding="utf-8", newline="") as f_out:
    writer = csv.writer(f_out)
    for line in lines:
        row = line.split(",", 1)
        writer.writerow(row)

print(f"Файл {output_file} успешно создан.")
