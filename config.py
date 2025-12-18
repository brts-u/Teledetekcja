"""
Konfiguracja parametrów algorytmu wykrywania torów kolejowych.

Wszystkie parametry można edytować w tym pliku.
"""

# ============ ŚCIEŻKI PLIKÓW ============
# Ścieżka do pliku wejściowego z danymi wielospektralnymi
INPUT_RASTER_PATH = r"C:\Users\User\OneDrive - Politechnika Warszawska\3 rok\Teledetekcja\projekt_3\grupa_6.tif"

# Ścieżka do pliku ze statystykami (GeoJSON z wartościami spektralnymi torów)
STATS_GEOJSON_PATH = "stats_grupa_6.geojson"

# ============ PROGI FILTRACJI SPEKTRALNEJ ============
# Progi dla ścisłej filtracji (wysoka pewność - core)
STRICT_SCALE_MIN = 0.90  # Mnożnik dla min wartości
STRICT_SCALE_MAX = 1.10  # Mnożnik dla max wartości

# Progi dla luźniejszej filtracji (rozszerzanie)
LOOSE_SCALE_MIN = 0.90
LOOSE_SCALE_MAX = 1.10

# ============ PROGI NDVI ============
# NDVI - Normalized Difference Vegetation Index
NDVI_MIN_STRICT = 0.21  # Ścisły próg NDVI
NDVI_MAX_STRICT = 0.61
NDVI_MIN_LOOSE = 0.21   # Luźny próg NDVI
NDVI_MAX_LOOSE = 0.61

# ============ PARAMETRY KLASTERYZACJI ============
# Minimalny rozmiar klastra do rozszerzania (w pikselach)
MIN_CLUSTER_SIZE_FOR_EXPANSION = 100

# Maksymalna liczba iteracji rozszerzania
MAX_EXPANSION_ITERATIONS = 5

# Minimalna liczba pasm, które muszą pasować (z 9: 8 pasm + NDVI)
MIN_BANDS_MATCH_EXPAND = 9

# ============ PARAMETRY ANALIZY KĄTÓW ============
# Parametry analizy kątów dla sieci kolejowej
MIN_ANGLE_DEGREES = 120  # Minimalny kąt - tory nie mają ostrych zakrętów
MAX_ANGLE_DEVIATION = 60  # Maksymalne odchylenie od linii prostej (180 - MIN_ANGLE)
ANGLE_SAMPLE_STEP = 5     # Co ile pikseli próbkować kąt
MIN_PATH_LENGTH_FOR_ANGLE_CHECK = 20  # Minimalna długość ścieżki do sprawdzania kątów

# ============ PARAMETRY ROZSZERZANIA KOŃCÓW TORÓW ============
EXTENSION_SEARCH_RADIUS = 3   # Promień szukania następnego piksela
EXTENSION_MAX_DISTANCE = 0    # 0 = bez limitu - szukaj aż do końca rastra
EXTENSION_MIN_BANDS = 7       # Minimalna liczba pasujących pasm przy rozszerzaniu

# ============ PARAMETRY ŁĄCZENIA ŚCIEŻEK (Dijkstra) ============
MAX_CONNECTION_DISTANCE = 30  # Maksymalna odległość (w pikselach) do szukania połączeń
MIN_ENDPOINT_NEIGHBORS = 1    # Min sąsiadów żeby być punktem końcowym
MAX_ENDPOINT_NEIGHBORS = 2    # Max sąsiadów żeby być punktem końcowym
COST_THRESHOLD = 1.5          # Maksymalny średni koszt ścieżki
PATH_CONNECTION_ITERATIONS = 0  # Liczba iteracji łączenia (0 = wyłączone)
MIN_CLUSTER_DEPTH_FOR_CONNECTION = 80  # Minimalna głębokość klastra do łączenia

# ============ PARAMETRY DRUGIEJ FILTRACJI (przy łączeniu) ============
SECOND_PASS_SCALE_MIN = 0.75
SECOND_PASS_SCALE_MAX = 1.25
SECOND_PASS_NDVI_MIN = 0.10
SECOND_PASS_NDVI_MAX = 0.75
SECOND_PASS_MIN_BANDS_MATCH = 4

# ============ PARAMETRY GRUBOŚCI ŚCIEŻEK ============
MIN_DEPTH_FOR_VECTORIZATION = 130  # Minimalna głębokość klastra
MAX_CLUSTER_THICKNESS = 15         # Maksymalna średnia grubość (autostrady są grubsze)
MIN_CLUSTER_THICKNESS = 1          # Minimalna grubość (szum)

# ============ PARAMETRY SZEROKOŚCI TORÓW ============
EXPECTED_TRACK_WIDTH_MIN = 2       # Minimalna oczekiwana szerokość toru w pikselach
EXPECTED_TRACK_WIDTH_MAX = 6       # Maksymalna szerokość (autostrady mają 8+ px)
WIDTH_SAMPLE_STEP = 10             # Co ile pikseli próbkować szerokość
WIDTH_VARIANCE_THRESHOLD = 2.5     # Maksymalna wariancja szerokości
MIN_VALID_WIDTH_RATIO = 0.7        # Min. % próbek z prawidłową szerokością

# ============ PARAMETRY WYKRYWANIA RÓWNOLEGŁYCH LINII ============
PARALLEL_CHECK_DISTANCE = 50       # Dystans do sprawdzenia równoległości
PARALLEL_ANGLE_THRESHOLD = 15      # Max różnica kąta dla uznania za równoległe (stopnie)
MIN_DISTANCE_BETWEEN_TRACKS = 20   # Minimalna odległość między dwoma torami

# ============ PARAMETRY ŁĄCZENIA ODCINKÓW (druga wektoryzacja) ============
VECTOR_CONNECT_CONE_ANGLE = 45     # Kąt stożka szukania przedłużenia (stopnie)
VECTOR_CONNECT_MAX_DISTANCE = 150  # Maksymalna odległość szukania połączenia (piksele)
VECTOR_CONNECT_MIN_BANDS = 5       # Luźniejszy próg dla pikseli łączących
MAX_SHARP_ANGLE_AFTER_CONNECT = 80 # Maksymalny kąt zakrętu po połączeniu (stopnie)

# ============ PARAMETRY FILTROWANIA AUTOSTRAD ============
HIGHWAY_MIN_WIDTH = 8              # Autostrady mają min 8 pikseli szerokości
HIGHWAY_LOW_VARIANCE = 1.0         # Autostrady mają bardzo stałą szerokość
HIGHWAY_MIN_LENGTH = 200           # Autostrady są długie

# ============ PARAMETRY DLA SKUPISK MIEJSKICH ============
URBAN_CLUSTER_DENSITY_THRESHOLD = 0.3  # Gęstość pikseli w okolicy
URBAN_NEIGHBOR_RADIUS = 50         # Promień sprawdzania gęstości

# ============ FLAGI KONTROLNE ============
# Pomijanie tworzenia maski, jeśli jest już utworzona
SKIP_BOOLEAN_MASK = False  # Ustaw True po pierwszym uruchomieniu

