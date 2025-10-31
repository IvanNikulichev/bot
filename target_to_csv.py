import pandas as pd
import re


def convert_to_poi_csv(
    input_path: str = "cultural_objects_mnn.xlsx", output_path: str = "poi_csv.csv"
):
   
    # Читаем Excel-файл
    df = pd.read_excel(input_path)

    # Удаляем пустые строки
    df = df.dropna(how="all").reset_index(drop=True)

    # Оставляем только нужные столбцы
    required_columns = ["id", "title", "coordinate", "description"]
    df = df[required_columns]

    # Сохраняем в CSV
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Файл сохранён: {output_path}")
    print(f"Количество строк: {len(df)}")


convert_to_poi_csv()
