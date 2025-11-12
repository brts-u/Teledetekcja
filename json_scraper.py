import json

# Wczytaj dane z pliku
with open('tory_staty.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Wypisz properties pierwszego feature
properties = data['features'][0]['properties']

# Wydrukuj w czytelnym formacie
print(json.dumps(properties, indent=4, ensure_ascii=False))
