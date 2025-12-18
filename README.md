# Projekt Teledetekcji - Wykrywanie Torów Kolejowych

## Opis projektu

Projekt służy do automatycznego wykrywania i wektoryzacji torów kolejowych na podstawie wielospektralnego obrazu satelitarnego (8 pasm + NDVI).

## Struktura projektu

```
projekt_3/
│
├── 📄 main.py                  # Główny skrypt uruchomieniowy
├── ⚙️ config.py                # Konfiguracja parametrów (EDYTOWALNY)
│
├── 📦 Moduły podstawowe:
│   ├── clusters.py             # Klasa Cluster i funkcje klasteryzacji
│   ├── json_scraper.py         # Parsowanie statystyk z GeoJSON
│   └── io_utils.py             # Narzędzia wejścia/wyjścia (zapis plików)
│
├── 🔧 Moduły przetwarzania:
│   ├── mask_operations.py      # Operacje na maskach spektralnych
│   ├── path_analysis.py        # Analiza ścieżek (kąty, szerokość, walidacja)
│   ├── path_connection.py      # Łączenie ścieżek (algorytm Dijkstry)
│   └── vectorization.py        # Wektoryzacja klastrów do GeoJSON
│
├── 📊 Pliki wejściowe:
│   ├── grupa_6.tif             # Obraz wielospektralny (8 pasm)
│   └── stats_grupa_6.geojson   # Statystyki spektralne torów kolejowych
│
├── 📁 Pliki wyjściowe (generowane):
│   ├── strict_mask.tif         # Maska wysokiej pewności
│   ├── result_mask.tif         # Maska po rozszerzaniu
│   ├── result_mask_connected.tif # Maska po połączeniu segmentów
│   ├── confidence_map.tif      # Mapa pewności (0-9 pasm)
│   ├── depths.tif              # Mapa głębokości klastrów
│   ├── train_tracks_raw.geojson # Surowe wektory torów
│   └── train_tracks.geojson    # Finalne wektory torów kolejowych
│
├── 📝 Dokumentacja:
│   ├── README.md               # Ten plik
│   ├── requirements.txt        # Zależności Python
│   └── instrukcja.txt          # Instrukcja projektu
│
├── 🗃️ Inne:
│   ├── p_koncowy.py            # Stary monolityczny skrypt (archiwum)
│   ├── venv/                   # Wirtualne środowisko Python (Windows)
│   └── __pycache__/            # Cache Pythona
│
└── 📂 .git/                    # Repozytorium Git
```

## Instalacja

```bash
pip install -r requirements.txt
```

## Uruchomienie

```bash
python main.py
```

## Konfiguracja

Wszystkie parametry algorytmu znajdują się w pliku `config.py`. Można je edytować bez modyfikacji kodu:

### Ścieżki plików

```python
INPUT_RASTER_PATH = "sciezka/do/pliku.tif"  # Obraz wejściowy
STATS_GEOJSON_PATH = "stats_grupa_6.geojson"  # Statystyki spektralne
```

### Progi filtracji spektralnej

```python
STRICT_SCALE_MIN = 0.90  # Mnożnik dla min wartości
STRICT_SCALE_MAX = 1.10  # Mnożnik dla max wartości
NDVI_MIN_STRICT = 0.21   # Minimalny NDVI
NDVI_MAX_STRICT = 0.61   # Maksymalny NDVI
```

### Parametry analizy kątów

```python
MIN_ANGLE_DEGREES = 120   # Minimalny kąt (tory nie mają ostrych zakrętów)
ANGLE_SAMPLE_STEP = 5     # Co ile pikseli próbkować kąt
```

### Parametry szerokości torów

```python
EXPECTED_TRACK_WIDTH_MIN = 2   # Min. szerokość toru (px)
EXPECTED_TRACK_WIDTH_MAX = 6   # Max. szerokość (autostrady > 8 px)
WIDTH_VARIANCE_THRESHOLD = 2.5 # Max. wariancja szerokości
```

### Parametry wektoryzacji

```python
MIN_DEPTH_FOR_VECTORIZATION = 130  # Min. głębokość klastra
MAX_CLUSTER_THICKNESS = 15         # Max. grubość (odrzuca autostrady)
```

## Moduły

### `config.py`
Plik konfiguracyjny z wszystkimi parametrami algorytmu. Edytuj ten plik, aby dostosować działanie programu.

### `clusters.py`
- `Cluster` - klasa reprezentująca klaster pikseli
- `label_with_diagonals()` - etykietowanie z 8-connectivity
- `create_clusters()` - tworzenie klastrów z obrazu etykiet

### `io_utils.py`
- `safe_rasterio_write()` - bezpieczny zapis plików GeoTIFF
- `safe_geojson_write()` - bezpieczny zapis plików GeoJSON
- `load_raster_bands()` - wczytywanie pasm rastrowych

### `mask_operations.py`
- `create_confidence_mask()` - tworzenie maski spektralnej
- `expand_mask_from_endpoints()` - rozszerzanie maski od krawędzi
- `extend_track_in_direction()` - rozszerzanie toru w kierunku
- `extend_all_track_endpoints()` - rozszerzanie wszystkich końców torów

### `path_analysis.py`
- `calculate_angle()` - obliczanie kąta między punktami
- `get_direction_vector()` - wektor kierunku ścieżki
- `check_path_angles()` - sprawdzanie ostrych kątów
- `analyze_path_width()` - analiza szerokości wzdłuż ścieżki
- `is_valid_railway_width()` - walidacja szerokości toru
- `check_parallel_false_positives()` - wykrywanie fałszywych równoległych linii

### `path_connection.py`
- `find_endpoints()` - znajdowanie punktów końcowych
- `dijkstra_path_between_endpoints()` - ścieżka Dijkstry
- `connect_paths_dijkstra()` - łączenie ścieżek algorytmem Dijkstry
- `find_vectors_in_cone()` - szukanie połączeń w stożku
- `connect_clusters_with_pixels()` - łączenie klastrów przez piksele

### `vectorization.py`
- `TrackVectorizer` - klasa do wektoryzacji torów
  - `load_and_preprocess_mask()` - wczytanie i przetworzenie maski
  - `create_clusters_with_depth()` - tworzenie klastrów z głębokościami
  - `filter_and_vectorize()` - filtracja i wektoryzacja
  - `connect_segments()` - łączenie segmentów
  - `final_vectorization()` - końcowa wektoryzacja

## Algorytm

### KROK 1: Pierwsza filtracja (wysokiej pewności)

Na podstawie statystyk spektralnych z pliku GeoJSON tworzona jest ścisła maska pikseli:

```
Dla każdego z 8 pasm spektralnych:
  piksel ∈ [min * 0.90, max * 1.10]
ORAZ
  NDVI ∈ [0.21, 0.61]
```

### KROK 2: Klasteryzacja i analiza kątów

1. Grupowanie pikseli w klastry (8-connectivity)
2. Obliczanie **głębokości** (najdłuższa ścieżka przez klaster)
3. Obliczanie **grubości** (powierzchnia / głębokość)
4. **Filtracja kątów** - odrzucenie klastrów z kątami < 120°

### KROK 3: Rozszerzanie od końców torów

1. Znajdź **końce prawidłowych torów**
2. Oblicz **kierunek przedłużenia** (wektor z ostatnich 10-20 pikseli)
3. Szukaj następnego piksela w stożku ±60° od kierunku
4. **Bez limitu odległości** - szukaj aż do końca rastra

### KROK 4: Filtrowanie według szerokości

Tory kolejowe mają **stałą szerokość** (2-6 pikseli), autostrady są szersze (8+).

### KROK 5: Wektoryzacja

Konwersja klastrów do GeoJSON jako LineString.

## Pliki wyjściowe

| Plik | Opis |
|------|------|
| `strict_mask.tif` | Maska wysokiej pewności (pierwsza filtracja) |
| `result_mask.tif` | Finalna maska po rozszerzaniu |
| `result_mask_connected.tif` | Maska po połączeniu segmentów |
| `confidence_map.tif` | Mapa pewności (ile pasm pasuje: 0-9) |
| `depths.tif` | Mapa głębokości klastrów |
| `train_tracks_raw.geojson` | Surowe zwektoryzowane tory |
| `train_tracks.geojson` | Finalne zwektoryzowane tory kolejowe |

## Wymagania

- Python 3.8+
- rasterio
- numpy
- geopandas
- shapely
- scikit-image
- scipy
- networkx

## Autorzy

Projekt Teledetekcji - Politechnika Warszawska, 2025

