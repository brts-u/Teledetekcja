# Projekt Teledetekcji - Wykrywanie Torów Kolejowych

## Opis projektu

Projekt służy do automatycznego wykrywania i wektoryzacji torów kolejowych na podstawie wielospektralnego obrazu satelitarnego (8 pasm + NDVI).

## Podejście algorytmiczne

### Założenia

1. **Tory kolejowe są długie** - ciągną się przez cały raster (1-4 główne linie)
2. **Zajezdnie są na skrajach** - nie w środku rastra, więc nie ma potrzeby ograniczania długości szukania
3. **Tory nie mają ostrych zakrętów** - minimalny kąt to ~120° (łagodne łuki)
4. **Tory są wąskie** - grubość 1-15 pikseli (autostrady są grubsze)

### Algorytm

#### KROK 1: Pierwsza filtracja (wysokiej pewności)

Na podstawie statystyk spektralnych z pliku GeoJSON tworzona jest ścisła maska pikseli:

```
Dla każdego z 8 pasm spektralnych:
  piksel ∈ [min * 0.90, max * 1.10]
ORAZ
  NDVI ∈ [0.21, 0.61]
```

#### KROK 2: Klasteryzacja i analiza kątów

1. Grupowanie pikseli w klastry (8-connectivity)
2. Obliczanie **głębokości** (najdłuższa ścieżka przez klaster)
3. Obliczanie **grubości** (powierzchnia / głębokość)
4. **Filtracja kątów** - odrzucenie klastrów z kątami < 120°

```python
# Próbkowanie kątów co 5 pikseli wzdłuż ścieżki
for i in range(5, len(path) - 5, 5):
    angle = calculate_angle(path[i-5], path[i], path[i+5])
    if angle < 120:
        reject_cluster()
```

#### KROK 3: Rozszerzanie od końców torów

**Kluczowa innowacja:** Zamiast rozszerzać od wszystkich pikseli:

1. Znajdź **końce prawidłowych torów**
2. Oblicz **kierunek przedłużenia** (wektor z ostatnich 10-20 pikseli)
3. Szukaj następnego piksela:
   - W stożku ±60° od aktualnego kierunku
   - Spełniającego progi spektralne (7/9 pasm)
   - Preferuj piksele "w linii" (najwyższy iloczyn skalarny)
4. **Bez limitu odległości** - szukaj aż do końca rastra lub braku pasujących pikseli

```python
# Płynna aktualizacja kierunku
current_dir = 0.7 * new_dir + 0.3 * current_dir  # Wygładzanie
```

#### KROK 4: Drugie przejście

Ponowna klasteryzacja i rozszerzanie z bardziej wymagającymi parametrami (8/9 pasm).

#### KROK 5: Wektoryzacja

Konwersja klastrów do GeoJSON jako LineString:
- Minimalna głębokość: 130 pikseli
- Grubość: 1-15 (odrzuca szum i autostrady)

## Parametry konfiguracyjne

```python
# Progi spektralne
STRICT_SCALE_MIN = 0.90
STRICT_SCALE_MAX = 1.10
NDVI_MIN = 0.21
NDVI_MAX = 0.61

# Analiza kątów
MIN_ANGLE_DEGREES = 120      # Minimalny kąt (tory nie mają ostrych zakrętów)
MAX_ANGLE_DEVIATION = 60     # Max odchylenie od kierunku przy rozszerzaniu
ANGLE_SAMPLE_STEP = 5        # Co ile pikseli próbkować kąt

# Rozszerzanie
EXTENSION_SEARCH_RADIUS = 3  # Promień szukania następnego piksela
EXTENSION_MAX_DISTANCE = 0   # 0 = bez limitu (szukaj aż do końca rastra)
EXTENSION_MIN_BANDS = 7      # Min. pasujących pasm (z 9)

# Wektoryzacja
MIN_DEPTH_FOR_VECTORIZATION = 130
MAX_CLUSTER_THICKNESS = 15   # Odrzuca autostrady
MIN_CLUSTER_THICKNESS = 1    # Odrzuca szum
```

## Pliki wejściowe

- `grupa_6.tif` - wielospektralny obraz satelitarny (8 pasm)
- `stats_grupa_6.geojson` - statystyki spektralne torów kolejowych

## Pliki wyjściowe

- `strict_mask.tif` - maska wysokiej pewności (pierwsza filtracja)
- `result_mask.tif` - finalna maska po rozszerzaniu
- `confidence_map.tif` - mapa pewności (ile pasm pasuje: 0-9)
- `depths.tif` - mapa głębokości klastrów
- `train_tracks.geojson` - zwektoryzowane tory kolejowe

## Uruchomienie

```bash
python p_koncowy.py
```

## Wymagania

- Python 3.8+
- rasterio
- numpy
- geopandas
- shapely
- scikit-image
- scipy

## Struktura projektu

```
projekt_3/
├── p_koncowy.py          # Główny skrypt
├── clusters.py           # Klasy klastrów i funkcje pomocnicze
├── json_scraper.py       # Parsowanie statystyk z GeoJSON
├── grupa_6.tif           # Obraz wejściowy
├── stats_grupa_6.geojson # Statystyki spektralne
└── README.md             # Dokumentacja
```

## Autorzy

Projekt Teledetekcji - Politechnika Warszawska, 2025

