from geo_utils import find_street

# Примеры запросов
queries = [
    "2 часа, от пл. Минина, модерн и кофейни",
    "56.328, 44.005; 1.5 часа; хочу стрит-арт и виды",
    "от ул. Большая Покровская",
    "ул. Покровская, 10",
    "Большая Покровская",
    "пл. Минина",
    "улица Покровская",
    "площадь Минина",
]

for q in queries:
    street, house = find_street(q)
    print(f"Query: '{q}' -> Street: '{street}', House: '{house}'")
