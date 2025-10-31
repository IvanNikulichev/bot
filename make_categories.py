import csv

input_file = "Pio.txt"   # замени на имя твоего файла, например: "categories.txt"
output_file = "catigories.csv" # имя результирующего CSV

with open(input_file, "r", encoding="utf-8") as f_in:
    # читаем все строки и убираем пустые/лишние пробелы
    lines = [line.strip() for line in f_in if line.strip()]

# Пишем в CSV
with open(output_file, "w", encoding="utf-8", newline="") as f_out:
    writer = csv.writer(f_out)
    for line in lines:
        # разделяем по первой запятой (на случай, если в категории есть запятые — но у тебя их нет)
        row = line.split(",", 1)
        writer.writerow(row)

print(f"✅ Файл {output_file} успешно создан.")