from geopandas import GeoDataFrame
from skimage.morphology import closing, opening, square
from shapely.geometry import LineString
import os
import heapq
import math
from clusters import *
import datetime

path = r"C:\Users\User\OneDrive - Politechnika Warszawska\3 rok\Teledetekcja\projekt_3\grupa_6.tif"


def safe_rasterio_write(filepath, data, height, width, crs, transform, count=1, dtype='uint8'):
    """
    Bezpiecznie zapisuje plik rasterowy. Jeśli plik jest zablokowany,
    próbuje zapisać pod alternatywną nazwą.
    """
    try:
        with rasterio.open(filepath, 'w', driver='GTiff', height=height, width=width,
                           crs=crs, transform=transform, count=count, dtype=dtype) as dst:
            dst.write(data, 1)
        print(f"  Zapisano: {filepath}")
        return filepath
    except Exception as e:
        # Plik zablokowany - użyj alternatywnej nazwy z timestampem
        base, ext = os.path.splitext(filepath)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_filepath = f"{base}_{timestamp}{ext}"
        print(f"  Uwaga: Nie można zapisać {filepath} ({e})")
        print(f"  Zapisuję jako: {alt_filepath}")
        with rasterio.open(alt_filepath, 'w', driver='GTiff', height=height, width=width,
                           crs=crs, transform=transform, count=count, dtype=dtype) as dst:
            dst.write(data, 1)
        return alt_filepath

# ============ PARAMETRY FILTRACJI ============
# Progi dla ścisłej filtracji (wysoka pewność - core)
STRICT_SCALE_MIN = 0.90  # Mnożnik dla min wartości
STRICT_SCALE_MAX = 1.10  # Mnożnik dla max wartości

# Progi dla luźniejszej filtracji (rozszerzanie) - teraz takie same jak strict
LOOSE_SCALE_MIN = 0.90   # Takie same jak strict dla kontrolowanego rozszerzania
LOOSE_SCALE_MAX = 1.10   # Takie same jak strict

# Progi NDVI - oryginalne wartości z projektu
NDVI_MIN_STRICT = 0.21   # Ścisły próg NDVI
NDVI_MAX_STRICT = 0.61
NDVI_MIN_LOOSE = 0.21    # Takie same dla kontroli
NDVI_MAX_LOOSE = 0.61

# Minimalny rozmiar klastra do rozszerzania
MIN_CLUSTER_SIZE_FOR_EXPANSION = 100

# Maksymalna liczba iteracji rozszerzania
MAX_EXPANSION_ITERATIONS = 5  # Tylko kilka iteracji

# Minimalna liczba pasm, które muszą pasować (z 9: 8 pasm + NDVI)
MIN_BANDS_MATCH_EXPAND = 9  # Dla rozszerzania - wymagane WSZYSTKIE 9 pasm

# Parametry analizy kątów dla sieci kolejowej
MIN_ANGLE_DEGREES = 120  # Minimalny kąt - tory nie mają ostrych zakrętów
MAX_ANGLE_DEVIATION = 60  # Maksymalne odchylenie od linii prostej (180 - MIN_ANGLE)
ANGLE_SAMPLE_STEP = 5    # Co ile pikseli próbkować kąt
MIN_PATH_LENGTH_FOR_ANGLE_CHECK = 20  # Minimalna długość ścieżki do sprawdzania kątów

# Parametry rozszerzania końców torów
EXTENSION_SEARCH_RADIUS = 3   # Promień szukania następnego piksela
EXTENSION_MAX_DISTANCE = 0    # 0 = bez limitu - szukaj aż do końca rastra lub braku pasujących pikseli
EXTENSION_MIN_BANDS = 7       # Minimalna liczba pasujących pasm przy rozszerzaniu

# Parametry łączenia ścieżek (Dijkstra)
MAX_CONNECTION_DISTANCE = 30  # Maksymalna odległość (w pikselach) do szukania połączeń (zmniejszona z 50)
MIN_ENDPOINT_NEIGHBORS = 1    # Min sąsiadów żeby być punktem końcowym
MAX_ENDPOINT_NEIGHBORS = 2    # Max sąsiadów żeby być punktem końcowym
COST_THRESHOLD = 1.5          # Maksymalny średni koszt ścieżki (zaostrzony z 2.5)
PATH_CONNECTION_ITERATIONS = 0  # TYMCZASOWO WYŁĄCZONE - ustawić na 3 po testach
MIN_CLUSTER_DEPTH_FOR_CONNECTION = 80  # Minimalna głębokość klastra do łączenia (zwiększone z 50)

# Dodatkowe parametry dla drugiej filtracji (luźniejsze progi przy łączeniu)
SECOND_PASS_SCALE_MIN = 0.75   # Jeszcze luźniejszy próg dla łączenia ścieżek
SECOND_PASS_SCALE_MAX = 1.25
SECOND_PASS_NDVI_MIN = 0.10    # Bardzo luźny NDVI dla łączenia
SECOND_PASS_NDVI_MAX = 0.75
SECOND_PASS_MIN_BANDS_MATCH = 4  # Mniej wymagane pasma (z 9)

# Parametry grubości ścieżek (filtrowanie autostrad)
MIN_DEPTH_FOR_VECTORIZATION = 130  # Minimalna głębokość klastra
MAX_CLUSTER_THICKNESS = 15         # Maksymalna średnia grubość (autostrady są grubsze)
MIN_CLUSTER_THICKNESS = 1          # Minimalna grubość (szum)

# Parametry szerokości torów (do filtrowania autostrad i fałszywych wykryć)
EXPECTED_TRACK_WIDTH_MIN = 2       # Minimalna oczekiwana szerokość toru w pikselach
EXPECTED_TRACK_WIDTH_MAX = 6       # Zmniejszone z 8 - autostrady mają 8+ px
WIDTH_SAMPLE_STEP = 10             # Co ile pikseli próbkować szerokość
WIDTH_VARIANCE_THRESHOLD = 2.5     # Zmniejszone z 3.0 - tory mają bardzo stałą szerokość
MIN_VALID_WIDTH_RATIO = 0.7        # Min. % próbek z prawidłową szerokością

# Parametry wykrywania równoległych fałszywych linii
PARALLEL_CHECK_DISTANCE = 50       # Dystans do sprawdzenia równoległości
PARALLEL_ANGLE_THRESHOLD = 15      # Max różnica kąta dla uznania za równoległe (stopnie)
MIN_DISTANCE_BETWEEN_TRACKS = 20   # Minimalna odległość między dwoma torami (pola to większa odległość)

# Parametry łączenia odcinków (druga wektoryzacja)
VECTOR_CONNECT_CONE_ANGLE = 45     # Kąt stożka szukania przedłużenia (stopnie)
VECTOR_CONNECT_MAX_DISTANCE = 150  # Maksymalna odległość szukania połączenia (piksele)
VECTOR_CONNECT_MIN_BANDS = 5       # Luźniejszy próg dla pikseli łączących (z 9)
MAX_SHARP_ANGLE_AFTER_CONNECT = 80 # Maksymalny kąt zakrętu po połączeniu (stopnie) - odrzuć jeśli ostrzejszy

# Parametry filtrowania autostrad (bardziej restrykcyjne)
HIGHWAY_MIN_WIDTH = 8              # Autostrady mają min 8 pikseli szerokości
HIGHWAY_LOW_VARIANCE = 1.0         # Autostrady mają bardzo stałą szerokość (niską wariancję)
HIGHWAY_MIN_LENGTH = 200           # Autostrady są długie

# Parametry dla skupisk miejskich
URBAN_CLUSTER_DENSITY_THRESHOLD = 0.3  # Gęstość pikseli w okolicy (wysoka = miasto)
URBAN_NEIGHBOR_RADIUS = 50         # Promień sprawdzania gęstości
# =============================================

print("Reading image bands...")
with rasterio.open(path) as src:
    cb = src.read(1).astype(float)  # pasmo coastal_blue
    b = src.read(2).astype(float)  # pasmo blue
    gi = src.read(3).astype(float)  # pasmo green_i
    g = src.read(4).astype(float)  # pasmo green
    y = src.read(5).astype(float)  # pasmo yellow
    r = src.read(6).astype(float)  # pasmo red
    re = src.read(7).astype(float)  # pasmo red_edge
    nir = src.read(8).astype(float)  # pasmo near_infra_red
    transform = src.transform
    crs = src.crs

# Wszystkie pasma w jednej tablicy dla łatwiejszego dostępu
all_bands = np.stack([cb, b, gi, g, y, r, re, nir], axis=0)

# Oblicz NDVI raz
ndvi = (nir - r) / (nir + r + 1e-6)


def create_confidence_mask(bands, arr, scale_min, scale_max, ndvi, ndvi_min, ndvi_max):
    """
    Tworzy maskę na podstawie progów pewności dla wszystkich pasm.
    Zwraca maskę boolean oraz mapę pewności (ile pasm pasuje).
    """
    height, width = bands[0].shape
    result_mask = np.ones((height, width), dtype=bool)
    confidence_map = np.zeros((height, width), dtype=np.int32)

    for i, (band, (min_val, mean_val, max_val)) in enumerate(zip(bands, arr)):
        min_thresh = min_val * scale_min
        max_thresh = max_val * scale_max
        band_mask = (band >= min_thresh) & (band <= max_thresh)
        result_mask &= band_mask
        confidence_map += band_mask.astype(np.int32)

    # Dodaj warunek NDVI
    ndvi_mask = (ndvi >= ndvi_min) & (ndvi <= ndvi_max)
    result_mask &= ndvi_mask
    confidence_map += ndvi_mask.astype(np.int32)

    return result_mask, confidence_map


def check_pixel_threshold(row, col, bands, arr, scale_min, scale_max, ndvi, ndvi_min, ndvi_max, min_bands_match=6):
    """
    Sprawdza czy piksel spełnia próg pewności.
    Zwraca True jeśli co najmniej min_bands_match pasm pasuje.
    """
    if row < 0 or row >= bands.shape[1] or col < 0 or col >= bands.shape[2]:
        return False

    matches = 0
    for i, (min_val, mean_val, max_val) in enumerate(arr):
        min_thresh = min_val * scale_min
        max_thresh = max_val * scale_max
        pixel_val = bands[i, row, col]
        if min_thresh <= pixel_val <= max_thresh:
            matches += 1

    # Sprawdź NDVI
    if ndvi_min <= ndvi[row, col] <= ndvi_max:
        matches += 1

    return matches >= min_bands_match


def calculate_pixel_cost(row, col, bands, arr, ndvi, ndvi_min, ndvi_max, loose_mode=False):
    """
    Oblicza koszt piksela - im bardziej pasuje do wzorca torów, tym niższy koszt.
    Zwraca wartość od 0 (idealnie pasuje) do wysokiej wartości (nie pasuje).
    loose_mode=True używa łagodniejszej oceny.
    """
    if row < 0 or row >= bands.shape[1] or col < 0 or col >= bands.shape[2]:
        return float('inf')

    total_deviation = 0.0

    # W trybie loose, rozszerzamy akceptowalny zakres
    range_multiplier = 1.5 if loose_mode else 1.0

    for i, (min_val, mean_val, max_val) in enumerate(arr):
        pixel_val = bands[i, row, col]
        # Oblicz odchylenie od średniej, znormalizowane przez zakres
        range_val = max(max_val - min_val, 1e-6) * range_multiplier

        if pixel_val < min_val:
            deviation = (min_val - pixel_val) / range_val
        elif pixel_val > max_val:
            deviation = (pixel_val - max_val) / range_val
        else:
            deviation = 0.0

        total_deviation += deviation

    # Sprawdź NDVI
    ndvi_val = ndvi[row, col]
    ndvi_range = max(ndvi_max - ndvi_min, 0.1) * range_multiplier
    if ndvi_val < ndvi_min:
        total_deviation += (ndvi_min - ndvi_val) / ndvi_range
    elif ndvi_val > ndvi_max:
        total_deviation += (ndvi_val - ndvi_max) / ndvi_range

    return total_deviation


def find_endpoints(mask, min_neighbors=1, max_neighbors=2):
    """
    Znajduje punkty końcowe ścieżek w masce.
    Punkt końcowy to piksel w masce, który ma 1-2 sąsiadów w masce.
    """
    height, width = mask.shape
    endpoints = []

    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1),          (0, 1),
                         (1, -1),  (1, 0), (1, 1)]

    white_pixels = np.argwhere(mask)

    for row, col in white_pixels:
        neighbor_count = 0
        for dr, dc in neighbors_offsets:
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc]:
                neighbor_count += 1

        if min_neighbors <= neighbor_count <= max_neighbors:
            endpoints.append((row, col))

    return endpoints


def dijkstra_path_between_endpoints(start, end, mask, bands, arr, ndvi, ndvi_min, ndvi_max, max_distance, loose_mode=False):
    """
    Znajduje najkrótszą ścieżkę między dwoma punktami końcowymi używając algorytmu Dijkstry.
    Ścieżka może przechodzić przez piksele spoza maski, ale preferuje podobne spektralnie.
    Zwraca (ścieżkę, średni koszt) lub (None, inf) jeśli nie znaleziono.
    loose_mode=True używa łagodniejszej oceny kosztów.
    """
    height, width = mask.shape
    start_row, start_col = start
    end_row, end_col = end

    # Sprawdź odległość euklidesową
    euclidean_dist = np.sqrt((end_row - start_row)**2 + (end_col - start_col)**2)
    if euclidean_dist > max_distance:
        return None, float('inf')

    # Sąsiedzi 8-kierunkowe
    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1),          (0, 1),
                         (1, -1),  (1, 0), (1, 1)]

    # Dijkstra
    heap = [(0.0, start_row, start_col, [(start_row, start_col)])]
    visited = set()
    costs = {(start_row, start_col): 0.0}

    while heap:
        current_cost, row, col, path = heapq.heappop(heap)

        if (row, col) in visited:
            continue
        visited.add((row, col))

        # Sprawdź czy dotarliśmy do celu
        if row == end_row and col == end_col:
            avg_cost = current_cost / max(len(path), 1)
            return path, avg_cost

        # Ogranicz obszar przeszukiwania
        if len(path) > max_distance * 2:
            continue

        for dr, dc in neighbors_offsets:
            nr, nc = row + dr, col + dc

            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if (nr, nc) in visited:
                continue

            # Koszt ruchu po przekątnej jest większy
            move_cost = 1.414 if dr != 0 and dc != 0 else 1.0

            # Oblicz koszt piksela docelowego
            if mask[nr, nc]:
                # Piksel już w masce - bardzo niski koszt
                pixel_cost = 0.1
            else:
                pixel_cost = calculate_pixel_cost(nr, nc, bands, arr, ndvi, ndvi_min, ndvi_max, loose_mode=loose_mode)
                # Dodaj karę za oddalanie się od linii prostej do celu (mniejsza w loose_mode)
                dist_to_end = np.sqrt((end_row - nr)**2 + (end_col - nc)**2)
                penalty = 0.005 if loose_mode else 0.01
                pixel_cost += dist_to_end * penalty

            new_cost = current_cost + move_cost * (1.0 + pixel_cost)

            if (nr, nc) not in costs or new_cost < costs[(nr, nc)]:
                costs[(nr, nc)] = new_cost
                new_path = path + [(nr, nc)]
                heapq.heappush(heap, (new_cost, nr, nc, new_path))

    return None, float('inf')


def connect_paths_dijkstra(mask, bands, arr, ndvi, ndvi_min, ndvi_max,
                           max_distance=50, cost_threshold=5.0, min_cluster_depth=50, loose_mode=False):
    """
    Łączy ścieżki poprzez wyszukiwanie połączeń między punktami końcowymi.
    Używa algorytmu Dijkstry do znajdowania optymalnych ścieżek.
    loose_mode=True używa łagodniejszych progów.
    """
    print("  Znajdowanie punktów końcowych...")
    endpoints = find_endpoints(mask, min_neighbors=1, max_neighbors=2)
    print(f"  Znaleziono {len(endpoints)} punktów końcowych")

    if len(endpoints) < 2:
        return mask, 0

    # Znajdź klastry i ich głębokości do filtrowania małych fragmentów
    labeled_img, num_features = label_with_diagonals(mask)
    cluster_depths = {}

    # Przypisz każdy endpoint do klastra
    endpoint_clusters = {}
    for ep in endpoints:
        cluster_id = labeled_img[ep[0], ep[1]]
        endpoint_clusters[ep] = cluster_id
        if cluster_id not in cluster_depths:
            cluster_depths[cluster_id] = np.sum(labeled_img == cluster_id)

    # Filtruj endpointy z małych klastrów (w loose_mode mniejszy próg)
    min_depth = min_cluster_depth // 2 if loose_mode else min_cluster_depth
    filtered_endpoints = [ep for ep in endpoints
                         if cluster_depths.get(endpoint_clusters[ep], 0) >= min_depth]
    print(f"  Po filtracji: {len(filtered_endpoints)} punktów końcowych")

    if len(filtered_endpoints) < 2:
        return mask, 0

    expanded_mask = mask.copy()
    connections_made = 0

    filtered_endpoints.sort(key=lambda x: (x[0], x[1]))

    # W loose_mode wyższy próg kosztu
    effective_threshold = cost_threshold * 1.5 if loose_mode else cost_threshold

    for i, ep1 in enumerate(filtered_endpoints):
        cluster1 = endpoint_clusters[ep1]

        for ep2 in filtered_endpoints[i+1:]:
            cluster2 = endpoint_clusters[ep2]

            if cluster1 == cluster2:
                continue

            dist = np.sqrt((ep2[0] - ep1[0])**2 + (ep2[1] - ep1[1])**2)
            if dist > max_distance:
                continue

            path, avg_cost = dijkstra_path_between_endpoints(
                ep1, ep2, expanded_mask, bands, arr, ndvi, ndvi_min, ndvi_max, max_distance, loose_mode=loose_mode
            )

            if path is not None and avg_cost < effective_threshold:
                for row, col in path:
                    expanded_mask[row, col] = True
                connections_made += 1
                print(f"    Połączono: {ep1} -> {ep2} (koszt: {avg_cost:.2f}, długość: {len(path)})")

    added_pixels = np.sum(expanded_mask) - np.sum(mask)
    print(f"  Utworzono {connections_made} połączeń, dodano {added_pixels} pikseli")

    return expanded_mask, connections_made


# ============ FUNKCJE ANALIZY KĄTÓW I ROZSZERZANIA TORÓW ============

def calculate_angle(p1, p2, p3):
    """
    Oblicza kąt w punkcie p2 utworzony przez punkty p1-p2-p3.
    Zwraca kąt w stopniach (0-180).
    """
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if mag1 == 0 or mag2 == 0:
        return 180.0

    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    angle = math.acos(cos_angle)
    return math.degrees(angle)


def get_direction_vector(path, from_end=True, sample_length=10):
    """
    Oblicza wektor kierunku na końcu ścieżki.
    from_end=True: kierunek na końcu ścieżki
    from_end=False: kierunek na początku ścieżki
    Zwraca znormalizowany wektor (dr, dc).
    """
    if len(path) < 2:
        return (0, 0)

    sample_length = min(sample_length, len(path) - 1)

    if from_end:
        p_end = path[-1]
        p_prev = path[-1 - sample_length]
        dr = p_end[0] - p_prev[0]
        dc = p_end[1] - p_prev[1]
    else:
        p_start = path[0]
        p_next = path[sample_length]
        dr = p_start[0] - p_next[0]
        dc = p_start[1] - p_next[1]

    mag = math.sqrt(dr**2 + dc**2)
    if mag == 0:
        return (0, 0)

    return (dr / mag, dc / mag)


def check_path_angles(path, min_angle=120, sample_step=5):
    """
    Sprawdza czy ścieżka nie zawiera ostrych kątów.
    Zwraca True jeśli ścieżka jest prawidłowa (bez ostrych kątów).
    """
    if len(path) < sample_step * 2 + 1:
        return True  # Za krótka do sprawdzenia

    for i in range(sample_step, len(path) - sample_step, sample_step):
        p1 = path[i - sample_step]
        p2 = path[i]
        p3 = path[i + sample_step]

        angle = calculate_angle(p1, p2, p3)
        if angle < min_angle:
            return False  # Znaleziono ostry kąt

    return True


def find_cluster_endpoints_with_direction(cluster, min_angle=120):
    """
    Znajduje punkty końcowe klastra wraz z ich kierunkami.
    Zwraca listę (punkt, kierunek, czy_prawidłowy).
    """
    path = cluster.longest_path
    if len(path) < 10:
        return []

    # Sprawdź kąty w całej ścieżce
    is_valid = check_path_angles(path, min_angle=min_angle)

    # Kierunek na początku (wychodzący)
    dir_start = get_direction_vector(path, from_end=False, sample_length=min(20, len(path)//4))
    # Kierunek na końcu (wychodzący)
    dir_end = get_direction_vector(path, from_end=True, sample_length=min(20, len(path)//4))

    endpoints = [
        (path[0], dir_start, is_valid),
        (path[-1], dir_end, is_valid)
    ]

    return endpoints


def extend_track_in_direction(start_point, direction, mask, bands, arr, scale_min, scale_max,
                               ndvi, ndvi_min, ndvi_max, max_distance=0, min_bands_match=7,
                               search_radius=3, min_angle=120):
    """
    Rozszerza tor w danym kierunku, szukając pikseli spełniających kryteria
    i utrzymując płynny kąt (bez ostrych zakrętów).
    max_distance=0 oznacza brak limitu - szukaj aż do końca rastra.
    Zwraca listę nowych pikseli do dodania do maski.
    """
    height, width = mask.shape
    new_pixels = []
    new_pixels_set = set()  # Dla szybszego sprawdzania

    current_pos = start_point
    current_dir = direction

    # Jeśli max_distance=0, ustaw na maksymalny wymiar rastra
    effective_max_distance = max_distance if max_distance > 0 else max(height, width)

    for step in range(effective_max_distance):
        # Szukaj najlepszego następnego piksela w kierunku current_dir
        best_pixel = None
        best_score = -float('inf')

        # Sprawdź piksele w stożku w kierunku current_dir
        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                if dr == 0 and dc == 0:
                    continue

                nr = int(current_pos[0] + dr)
                nc = int(current_pos[1] + dc)

                # Sprawdź granice
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue

                # Sprawdź czy już w masce lub nowych pikselach
                if mask[nr, nc] or (nr, nc) in new_pixels_set:
                    continue

                # Oblicz kąt między aktualnym kierunkiem a nowym
                move_dir = (dr, dc)
                move_mag = math.sqrt(dr**2 + dc**2)
                if move_mag == 0:
                    continue
                move_dir_norm = (dr / move_mag, dc / move_mag)

                # Iloczyn skalarny daje cosinus kąta
                dot = current_dir[0] * move_dir_norm[0] + current_dir[1] * move_dir_norm[1]

                # Preferuj piksele w kierunku ruchu (dot bliski 1)
                if dot < 0.5:  # Odrzuć piksele pod kątem > 60 stopni
                    continue

                # Sprawdź czy piksel spełnia próg spektralny
                if not check_pixel_threshold(nr, nc, bands, arr, scale_min, scale_max,
                                            ndvi, ndvi_min, ndvi_max, min_bands_match):
                    continue

                # Oblicz score: preferuj piksele w kierunku i bliskie spektralnie
                # dot jest od 0.5 do 1.0, więc score jest proporcjonalny do "prostości"
                score = dot

                if score > best_score:
                    best_score = score
                    best_pixel = (nr, nc)

        if best_pixel is None:
            break  # Nie znaleziono następnego piksela

        new_pixels.append(best_pixel)
        new_pixels_set.add(best_pixel)

        # Aktualizuj kierunek (uśrednij z poprzednim dla płynności)
        dr = best_pixel[0] - current_pos[0]
        dc = best_pixel[1] - current_pos[1]
        mag = math.sqrt(dr**2 + dc**2)
        if mag > 0:
            new_dir = (dr / mag, dc / mag)
            # Płynna aktualizacja kierunku (70% nowy, 30% stary)
            current_dir = (0.7 * new_dir[0] + 0.3 * current_dir[0],
                          0.7 * new_dir[1] + 0.3 * current_dir[1])
            # Renormalizuj
            mag = math.sqrt(current_dir[0]**2 + current_dir[1]**2)
            if mag > 0:
                current_dir = (current_dir[0] / mag, current_dir[1] / mag)

        current_pos = best_pixel

    return new_pixels


def extend_all_track_endpoints(mask, clusters, bands, arr, scale_min, scale_max,
                                ndvi, ndvi_min, ndvi_max, min_depth=100,
                                max_distance=100, min_bands_match=7, min_angle=120):
    """
    Rozszerza wszystkie prawidłowe tory od ich punktów końcowych.
    Pracuje tylko na dużych klastrach bez ostrych kątów.
    """
    extended_mask = mask.copy()
    total_added = 0
    tracks_extended = 0

    print(f"  Analizowanie {len(clusters)} klastrów...")

    for cluster in clusters:
        if cluster.depth < min_depth:
            continue

        # Znajdź punkty końcowe z kierunkami
        endpoints = find_cluster_endpoints_with_direction(cluster, min_angle=min_angle)

        for endpoint, direction, is_valid in endpoints:
            if not is_valid:
                continue  # Pomiń klastry z ostrymi kątami

            if direction == (0, 0):
                continue

            # Rozszerz w tym kierunku
            new_pixels = extend_track_in_direction(
                endpoint, direction, extended_mask, bands, arr,
                scale_min, scale_max, ndvi, ndvi_min, ndvi_max,
                max_distance=max_distance, min_bands_match=min_bands_match,
                min_angle=min_angle
            )

            if new_pixels:
                for r, c in new_pixels:
                    extended_mask[r, c] = True
                total_added += len(new_pixels)
                tracks_extended += 1

    print(f"  Rozszerzono {tracks_extended} końców torów, dodano {total_added} pikseli")
    return extended_mask


# ============ KONIEC FUNKCJI ANALIZY KĄTÓW ============


def expand_mask_from_endpoints(mask, bands, arr, scale_min, scale_max, ndvi, ndvi_min, ndvi_max,
                                min_cluster_size=50, max_iterations=100, min_bands_match=6):
    """
    Rozszerza maskę poprzez sprawdzanie sąsiadów na krawędziach dużych klastrów.
    Rozszerza tylko od klastrów o minimalnym rozmiarze.
    ZOPTYMALIZOWANA WERSJA - tylko jeden krok rozszerzania na iterację.
    """
    expanded_mask = mask.copy()
    height, width = mask.shape

    # 8-connectivity - sąsiedzi wliczając przekątne
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]

    # Znajdź duże klastry do rozszerzania
    labeled_img, num_features = label_with_diagonals(mask)
    cluster_sizes = {}
    for label_id in range(1, num_features + 1):
        cluster_sizes[label_id] = np.sum(labeled_img == label_id)

    # Maska tylko dużych klastrów
    large_cluster_mask = np.zeros_like(mask)
    for label_id, size in cluster_sizes.items():
        if size >= min_cluster_size:
            large_cluster_mask |= (labeled_img == label_id)

    large_cluster_count = np.sum(large_cluster_mask)
    num_large = sum(1 for s in cluster_sizes.values() if s >= min_cluster_size)
    print(f"  Rozszerzanie od {num_large} dużych klastrów ({large_cluster_count} pikseli)")

    total_added = 0

    for iteration in range(max_iterations):
        # Znajdź piksele na krawędziach (tylko 1 piksel od maski)
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(expanded_mask & large_cluster_mask)
        candidates = dilated & ~expanded_mask

        candidate_coords = np.argwhere(candidates)

        if len(candidate_coords) == 0:
            print(f"  Zatrzymano - brak kandydatów")
            break

        added_this_iteration = 0
        new_pixels = []

        for nr, nc in candidate_coords:
            # Sprawdź czy piksel spełnia próg
            if check_pixel_threshold(nr, nc, bands, arr, scale_min, scale_max,
                                    ndvi, ndvi_min, ndvi_max, min_bands_match):
                new_pixels.append((nr, nc))
                added_this_iteration += 1

        # Dodaj nowe piksele do maski
        for nr, nc in new_pixels:
            expanded_mask[nr, nc] = True
            large_cluster_mask[nr, nc] = True

        total_added += added_this_iteration
        print(f"  Iteracja {iteration + 1}: dodano {added_this_iteration} pikseli")

        if added_this_iteration == 0:
            print(f"  Zatrzymano - brak nowych pikseli do dodania")
            break

    print(f"Łącznie dodano {total_added} pikseli przez rozszerzanie")
    return expanded_mask


# Pomijanie tworzenia maski, jeśli jest już utworzona
SKIP_BOOLEAN_MASK = False  # Ustaw True po pierwszym uruchomieniu

# Sprawdź czy plik maski istnieje - jeśli nie, utwórz go niezależnie od flagi
if not SKIP_BOOLEAN_MASK or not os.path.exists('result_mask.tif'):
    from json_scraper import arr

    # KROK 1: Utwórz ścisłą maskę (wysoką pewność - "core" torów)
    print("Creating strict boolean mask (high confidence cores)...")
    strict_mask, strict_confidence = create_confidence_mask(
        [cb, b, gi, g, y, r, re, nir], arr,
        STRICT_SCALE_MIN, STRICT_SCALE_MAX,
        ndvi, NDVI_MIN_STRICT, NDVI_MAX_STRICT
    )

    strict_count = np.sum(strict_mask)
    print(f"  Ścisła maska: {strict_count} pikseli")

    # Zapisz ścisłą maskę do podglądu
    safe_rasterio_write('strict_mask.tif', strict_mask.astype('uint8'),
                        strict_mask.shape[0], strict_mask.shape[1], crs, transform)

    # KROK 2: Wstępna klasteryzacja i analiza kątów
    print("Performing initial clustering and angle analysis...")
    labeled_initial, num_initial = label_with_diagonals(strict_mask)
    print(f"  Znaleziono {num_initial} wstępnych klastrów")

    # Utwórz klastry do analizy kątów
    initial_clusters = create_clusters(labeled_initial)
    print(f"  Utworzono {len(initial_clusters)} klastrów")

    # Oblicz głębokość dla klastrów >= 50 pikseli
    print("  Obliczanie głębokości klastrów...")
    for cluster in initial_clusters.values():
        if cluster.size() >= 50:
            cluster.get_depth()

    # KROK 3: Rozszerzanie od końców prawidłowych torów (bez ostrych kątów)
    print("Extending tracks from valid endpoints (no sharp angles)...")
    result_mask = extend_all_track_endpoints(
        strict_mask,
        list(initial_clusters.values()),
        all_bands, arr,
        LOOSE_SCALE_MIN, LOOSE_SCALE_MAX,
        ndvi, NDVI_MIN_LOOSE, NDVI_MAX_LOOSE,
        min_depth=MIN_DEPTH_FOR_VECTORIZATION,  # Tylko długie tory
        max_distance=EXTENSION_MAX_DISTANCE,
        min_bands_match=EXTENSION_MIN_BANDS,
        min_angle=MIN_ANGLE_DEGREES
    )

    extended_count = np.sum(result_mask)
    print(f"  Po rozszerzeniu: {extended_count} pikseli (+{extended_count - strict_count})")

    # KROK 4: Ponowna klasteryzacja i drugie rozszerzenie
    print("Second pass: re-clustering and extending...")
    labeled_second, num_second = label_with_diagonals(result_mask)
    second_clusters = create_clusters(labeled_second)

    for cluster in second_clusters.values():
        if cluster.size() >= 50:
            cluster.get_depth()

    result_mask = extend_all_track_endpoints(
        result_mask,
        list(second_clusters.values()),
        all_bands, arr,
        LOOSE_SCALE_MIN, LOOSE_SCALE_MAX,
        ndvi, NDVI_MIN_LOOSE, NDVI_MAX_LOOSE,
        min_depth=100,  # Nieco mniejszy próg dla drugiego przejścia
        max_distance=EXTENSION_MAX_DISTANCE,  # Bez limitu (0)
        min_bands_match=EXTENSION_MIN_BANDS + 1,  # Bardziej wymagający
        min_angle=MIN_ANGLE_DEGREES
    )

    final_count = np.sum(result_mask)
    print(f"  Finalna maska: {final_count} pikseli (+{final_count - strict_count} od początku)")

    # Zapisz wynikowy raster
    safe_rasterio_write('result_mask.tif', result_mask.astype('uint8'),
                        result_mask.shape[0], result_mask.shape[1], crs, transform)

    # Zapisz mapę pewności do analizy
    safe_rasterio_write('confidence_map.tif', strict_confidence,
                        strict_confidence.shape[0], strict_confidence.shape[1],
                        crs, transform, dtype='int32')

else:
    print("Using created boolean mask...")

print("Performing morphological operations...")
with rasterio.open('result_mask.tif') as src:
    img = src.read(1).astype(bool)  # Read as boolean
assert img is not None, "file could not be read, check with os.path.exists()"
kernel3 = square(3)
closed_img = closing(img, kernel3)

kernel2 = square(2)  # Equivalent to a 2x2 kernel
opened_img = opening(closed_img, kernel2)

labeled_img, num_features = label_with_diagonals(img)

print("Creating clusters...")
t0 = time.time()
clusters = create_clusters(labeled_img)
t1 = time.time()
print(f"Total clusters found: {len(clusters)}, {t1 - t0:.2f} seconds")

last_percent = 0
size_sum = 0
for cluster in clusters:
    size_sum += clusters[cluster].size()
print('Calculating depths for clusters (this may take a while)...')
partial_size_sum = 0
t0 = time.time()
for cluster in clusters.values():
    if not cluster.size() < 100:
        depth = cluster.get_depth()
    partial_size_sum += cluster.size()
    percent = (partial_size_sum + 1) * 100 // size_sum  # Calculate progress percentage
    if percent > last_percent:  # Update only when the percentage changes
        elapsed = time.time() - t0
        sys.stdout.write(
            f"\rProgress: {percent}% (est. time remaining {elapsed * (100 - percent) / (percent + 1):.2f} seconds)")
        sys.stdout.flush()
        last_percent = percent

sys.stdout.write("\nDone!\n")  # Move to the next line after the loop

# save to tiff - użyj bezpiecznego zapisu
depth_raster = np.zeros(labeled_img.shape, dtype='float32')
for cluster in clusters.values():
    depth = cluster.depth
    for (row, col) in cluster.cells:
        depth_raster[row, col] = depth

safe_rasterio_write('depths.tif', depth_raster, labeled_img.shape[0], labeled_img.shape[1],
                    crs, transform, count=1, dtype='float32')

clusters = list(clusters.values())
clusters.sort(key=lambda c: c.depth, reverse=True)


def measure_width_at_point(point, direction, cluster_cells_set, max_width=30):
    """
    Mierzy szerokość klastra w punkcie prostopadle do kierunku ścieżki.
    Zwraca szerokość w pikselach.
    """
    row, col = point
    dy, dx = direction

    # Wektor prostopadły
    perp_dy, perp_dx = -dx, dy

    # Normalizuj
    length = math.sqrt(perp_dy**2 + perp_dx**2)
    if length < 1e-6:
        return 1
    perp_dy /= length
    perp_dx /= length

    # Szukaj w obie strony prostopadle do kierunku
    width = 1  # Sam punkt

    for direction_mult in [1, -1]:
        for dist in range(1, max_width):
            nr = int(round(row + direction_mult * dist * perp_dy))
            nc = int(round(col + direction_mult * dist * perp_dx))

            if (nr, nc) in cluster_cells_set:
                width += 1
            else:
                break

    return width


def analyze_path_width(path, cluster_cells, sample_step=10, direction_sample=5):
    """
    Analizuje szerokość klastra wzdłuż ścieżki.
    Zwraca: (średnia_szerokość, wariancja_szerokości, lista_szerokości)
    """
    if len(path) < direction_sample * 2 + 1:
        return 1.0, 0.0, [1]

    cluster_cells_set = set(cluster_cells)
    widths = []

    for i in range(direction_sample, len(path) - direction_sample, sample_step):
        # Oblicz lokalny kierunek
        p_prev = path[i - direction_sample]
        p_next = path[i + direction_sample]

        dy = p_next[0] - p_prev[0]
        dx = p_next[1] - p_prev[1]

        length = math.sqrt(dy**2 + dx**2)
        if length < 1e-6:
            continue

        direction = (dy / length, dx / length)

        # Zmierz szerokość w tym punkcie
        width = measure_width_at_point(path[i], direction, cluster_cells_set)
        widths.append(width)

    if not widths:
        return 1.0, 0.0, [1]

    avg_width = sum(widths) / len(widths)
    variance = sum((w - avg_width)**2 for w in widths) / len(widths)

    return avg_width, variance, widths


def is_valid_railway_width(cluster, min_width=2, max_width=8, max_variance=3.0, min_valid_ratio=0.7):
    """
    Sprawdza czy klaster ma szerokość charakterystyczną dla torów kolejowych.
    Tory mają stałą szerokość (niską wariancję) w zakresie 2-8 pikseli.
    Autostrady są szersze (10-20+ pikseli).

    Zwraca: (czy_prawidłowy, średnia_szerokość, wariancja, powód_odrzucenia)
    """
    path = cluster.longest_path
    if path is None or len(path) < 20:
        return True, 0, 0, None  # Za krótkie do analizy

    avg_width, variance, widths = analyze_path_width(path, cluster.cells)

    # Sprawdź ile próbek ma prawidłową szerokość
    valid_samples = sum(1 for w in widths if min_width <= w <= max_width)
    valid_ratio = valid_samples / len(widths) if widths else 0

    # Odrzuć jeśli za szeroki (autostrada)
    if avg_width > max_width:
        return False, avg_width, variance, f"za_szeroki ({avg_width:.1f} > {max_width})"

    # Odrzuć jeśli za duża wariancja (niestabilna szerokość - prawdopodobnie nie tor)
    if variance > max_variance and avg_width > 4:
        return False, avg_width, variance, f"niestabilna_szerokosc (var={variance:.1f})"

    # Odrzuć jeśli za mało próbek ma prawidłową szerokość
    if valid_ratio < min_valid_ratio:
        return False, avg_width, variance, f"za_malo_prawidlowych ({valid_ratio:.1%})"

    return True, avg_width, variance, None


def check_parallel_false_positives(clusters_list, min_distance=20, max_angle_diff=15):
    """
    Wykrywa pary równoległych klastrów które są blisko siebie -
    mogą być fałszywymi wykryciami (np. krawędzie pól).

    Zwraca listę klastrów do odrzucenia (krótszy z pary równoległych).
    """
    clusters_to_reject = set()

    valid_clusters = [c for c in clusters_list if c.depth >= MIN_DEPTH_FOR_VECTORIZATION
                      and c.longest_path is not None and len(c.longest_path) >= 20]

    for i, c1 in enumerate(valid_clusters):
        if id(c1) in clusters_to_reject:
            continue

        path1 = c1.longest_path
        # Oblicz kierunek klastra 1
        dir1 = get_direction_vector(path1, from_end=True, sample_length=min(50, len(path1)//2))

        for c2 in valid_clusters[i+1:]:
            if id(c2) in clusters_to_reject:
                continue

            path2 = c2.longest_path
            # Oblicz kierunek klastra 2
            dir2 = get_direction_vector(path2, from_end=True, sample_length=min(50, len(path2)//2))

            # Sprawdź czy kierunki są równoległe
            dot = abs(dir1[0]*dir2[0] + dir1[1]*dir2[1])
            if dot < math.cos(math.radians(max_angle_diff)):
                continue  # Nie są równoległe

            # Sprawdź odległość między klastrami
            min_dist = float('inf')
            # Próbkuj kilka punktów - osobne indeksy dla każdej ścieżki
            sample_indices1 = [0, len(path1)//4, len(path1)//2, 3*len(path1)//4, len(path1)-1]
            sample_indices2 = [0, len(path2)//4, len(path2)//2, 3*len(path2)//4, len(path2)-1]
            for idx1 in sample_indices1:
                if idx1 >= len(path1):
                    continue
                p1 = path1[idx1]
                for idx2 in sample_indices2:
                    if idx2 >= len(path2):
                        continue
                    p2 = path2[idx2]
                    dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                    min_dist = min(min_dist, dist)

            # Jeśli są blisko i równoległe - to podejrzane
            if min_dist < min_distance:
                # Odrzuć krótszy z dwóch
                if c1.depth < c2.depth:
                    clusters_to_reject.add(id(c1))
                else:
                    clusters_to_reject.add(id(c2))

    return clusters_to_reject


def calculate_cluster_thickness(cluster):
    """
    Oblicza średnią grubość klastra jako stosunek powierzchni do długości najdłuższej ścieżki.
    Cienkie linie (tory) mają niską grubość, grube obiekty (autostrady) mają wysoką.
    """
    if cluster.depth <= 0:
        return float('inf')

    # Grubość = powierzchnia / długość
    # Dla idealnej linii o szerokości 1px: thickness = depth / depth = 1
    # Dla grubszych obiektów: thickness > 1
    thickness = cluster.size() / cluster.depth
    return thickness


# Wektoryzacja
print("Starting vectorization...")
print(f"Filtering clusters: depth >= {MIN_DEPTH_FOR_VECTORIZATION}, thickness in [{MIN_CLUSTER_THICKNESS}, {MAX_CLUSTER_THICKNESS}]")
print(f"Width filter: {EXPECTED_TRACK_WIDTH_MIN}-{EXPECTED_TRACK_WIDTH_MAX} px, max variance {WIDTH_VARIANCE_THRESHOLD}")

# Najpierw wykryj równoległe fałszywe linie (np. krawędzie pól)
print("Checking for parallel false positives...")
parallel_rejects = check_parallel_false_positives(
    clusters,
    min_distance=MIN_DISTANCE_BETWEEN_TRACKS,
    max_angle_diff=PARALLEL_ANGLE_THRESHOLD
)
print(f"  Oznaczono {len(parallel_rejects)} klastrów jako potencjalne fałszywe (równoległe)")

data = {'geometry': [], 'depth': [], 'thickness': [], 'avg_width': [], 'width_variance': []}
skipped_thin = 0
skipped_thick = 0
skipped_short = 0
skipped_width = 0
skipped_parallel = 0

for cluster in clusters:
    # Jeśli depth < MIN_DEPTH to prawdopodobnie nie jest to obiekt, który nas obchodzi
    if cluster.depth < MIN_DEPTH_FOR_VECTORIZATION:
        skipped_short += 1
        continue

    # Sprawdź czy nie jest w zestawie równoległych fałszywych
    if id(cluster) in parallel_rejects:
        skipped_parallel += 1
        continue

    # Oblicz grubość klastra
    thickness = calculate_cluster_thickness(cluster)

    # Filtruj zbyt cienkie (szum) i zbyt grube (autostrady, budynki)
    if thickness < MIN_CLUSTER_THICKNESS:
        skipped_thin += 1
        continue

    if thickness > MAX_CLUSTER_THICKNESS:
        skipped_thick += 1
        continue

    # NOWY FILTR: Sprawdź szerokość wzdłuż ścieżki
    is_valid_width, avg_width, width_var, reject_reason = is_valid_railway_width(
        cluster,
        min_width=EXPECTED_TRACK_WIDTH_MIN,
        max_width=EXPECTED_TRACK_WIDTH_MAX,
        max_variance=WIDTH_VARIANCE_THRESHOLD,
        min_valid_ratio=MIN_VALID_WIDTH_RATIO
    )

    if not is_valid_width:
        skipped_width += 1
        print(f"    Odrzucono klaster (depth={cluster.depth}): {reject_reason}")
        continue

    full_path = cluster.longest_path
    steps = full_path[::10]
    geo_coords = []
    for row, col in steps:
        x, y = transform * (col, row)
        geo_coords.append((x, y))

    geo_coords_center = []
    for row, col in steps:
        x, y = transform * (col + 0.5, row + 0.5)
        geo_coords_center.append((x, y))

    line = LineString(geo_coords)
    data['geometry'].append(line)
    data['depth'].append(cluster.depth)
    data['thickness'].append(thickness)
    data['avg_width'].append(avg_width)
    data['width_variance'].append(width_var)

print(f"  Pominięto: {skipped_short} za krótkich, {skipped_thin} za cienkich, {skipped_thick} za grubych")
print(f"  Pominięto: {skipped_width} nieprawidłowa szerokość, {skipped_parallel} równoległe fałszywe")
print(f"  Zachowano: {len(data['geometry'])} klastrów")

gdf = GeoDataFrame(data, crs=crs)

# Bezpieczny zapis GeoJSON (pierwsza wektoryzacja)
geojson_path = 'train_tracks_raw.geojson'
try:
    gdf.to_file(geojson_path, driver='GeoJSON')
    print(f"  Zapisano: {geojson_path}")
except PermissionError:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    alt_path = f"train_tracks_raw_{timestamp}.geojson"
    print(f"  Zapisuję jako: {alt_path}")
    gdf.to_file(alt_path, driver='GeoJSON')


# ============ DRUGA WEKTORYZACJA - ŁĄCZENIE ODCINKÓW ============
print("\n" + "="*60)
print("DRUGA WEKTORYZACJA - Łączenie odcinków i filtrowanie")
print("="*60)

# Zbierz prawidłowe klastry do łączenia
valid_clusters_for_connection = []
for cluster in clusters:
    if cluster.depth < MIN_DEPTH_FOR_VECTORIZATION:
        continue
    thickness = calculate_cluster_thickness(cluster)
    if thickness < MIN_CLUSTER_THICKNESS or thickness > MAX_CLUSTER_THICKNESS:
        continue
    is_valid_width, avg_width, width_var, _ = is_valid_railway_width(
        cluster,
        min_width=EXPECTED_TRACK_WIDTH_MIN,
        max_width=EXPECTED_TRACK_WIDTH_MAX,
        max_variance=WIDTH_VARIANCE_THRESHOLD,
        min_valid_ratio=MIN_VALID_WIDTH_RATIO
    )
    if is_valid_width:
        valid_clusters_for_connection.append(cluster)

print(f"Klastrów do łączenia: {len(valid_clusters_for_connection)}")


def find_vectors_in_cone(endpoint, direction, all_clusters, current_cluster,
                         cone_angle=45, max_distance=150):
    """
    Szuka końców innych wektorów w stożku przed punktem końcowym.
    Zwraca listę (klaster, punkt_końcowy, odległość, kąt).
    """
    candidates = []
    cone_cos = math.cos(math.radians(cone_angle))

    for cluster in all_clusters:
        if cluster is current_cluster:
            continue

        path = cluster.longest_path
        if path is None or len(path) < 10:
            continue

        # Sprawdź oba końce ścieżki
        for end_idx in [0, -1]:
            other_end = path[end_idx]

            # Oblicz odległość
            dist = math.sqrt((other_end[0] - endpoint[0])**2 +
                           (other_end[1] - endpoint[1])**2)

            if dist > max_distance or dist < 5:  # Za daleko lub za blisko
                continue

            # Oblicz kierunek do drugiego końca
            dir_to_other = (
                (other_end[0] - endpoint[0]) / dist,
                (other_end[1] - endpoint[1]) / dist
            )

            # Sprawdź czy w stożku (iloczyn skalarny)
            dot = direction[0] * dir_to_other[0] + direction[1] * dir_to_other[1]

            if dot >= cone_cos:  # W stożku
                angle = math.degrees(math.acos(max(-1, min(1, dot))))
                candidates.append((cluster, other_end, dist, angle, end_idx))

    # Sortuj po odległości
    candidates.sort(key=lambda x: x[2])
    return candidates


def connect_clusters_with_pixels(start_point, end_point, mask, bands, arr, ndvi,
                                  ndvi_min, ndvi_max, min_bands_match=5):
    """
    Próbuje połączyć dwa punkty przez "zdatne" piksele.
    Używa prostego pathfindingu z luźniejszymi progami.
    Zwraca listę pikseli łączących lub None jeśli nie udało się połączyć.
    """
    height, width = mask.shape

    # Prosty A* lub liniowe przeszukiwanie
    path = []
    current = start_point
    target = end_point
    visited = {start_point}

    max_steps = int(math.sqrt((end_point[0]-start_point[0])**2 +
                              (end_point[1]-start_point[1])**2) * 2)

    for _ in range(max_steps):
        if current == target:
            return path

        # Znajdź najlepszego sąsiada w kierunku celu
        best_neighbor = None
        best_score = float('inf')

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                nr, nc = current[0] + dr, current[1] + dc

                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                if (nr, nc) in visited:
                    continue

                # Odległość do celu
                dist_to_target = math.sqrt((target[0]-nr)**2 + (target[1]-nc)**2)

                # Sprawdź czy piksel jest zdatny (luźniejszy próg)
                is_valid = check_pixel_threshold(nr, nc, bands, arr, 0.80, 1.20,
                                                 ndvi, ndvi_min, ndvi_max, min_bands_match)

                # Jeśli już w masce - świetnie
                if mask[nr, nc]:
                    score = dist_to_target - 100  # Bonus za bycie w masce
                elif is_valid:
                    score = dist_to_target
                else:
                    score = dist_to_target + 50  # Kara za niezdatny piksel

                if score < best_score:
                    best_score = score
                    best_neighbor = (nr, nc)

        if best_neighbor is None:
            return None  # Nie udało się

        visited.add(best_neighbor)
        if not mask[best_neighbor[0], best_neighbor[1]]:
            path.append(best_neighbor)
        current = best_neighbor

        # Sprawdź czy blisko celu
        if math.sqrt((target[0]-current[0])**2 + (target[1]-current[1])**2) < 3:
            return path

    return None


def check_connection_angle(cluster1, end_idx1, cluster2, end_idx2, max_angle=80):
    """
    Sprawdza czy połączenie dwóch klastrów nie tworzy zbyt ostrego kąta.
    Zwraca True jeśli kąt jest akceptowalny.
    """
    path1 = cluster1.longest_path
    path2 = cluster2.longest_path

    # Oblicz kierunek na końcu klastra 1
    sample_len = min(20, len(path1)//4)
    if end_idx1 == 0:
        dir1 = get_direction_vector(path1, from_end=False, sample_length=sample_len)
    else:
        dir1 = get_direction_vector(path1, from_end=True, sample_length=sample_len)

    # Oblicz kierunek na początku klastra 2 (w kierunku połączenia)
    sample_len = min(20, len(path2)//4)
    if end_idx2 == 0:
        # Kierunek wchodzący do klastra 2 (odwrotny)
        dir2_raw = get_direction_vector(path2, from_end=False, sample_length=sample_len)
        dir2 = (-dir2_raw[0], -dir2_raw[1])
    else:
        dir2_raw = get_direction_vector(path2, from_end=True, sample_length=sample_len)
        dir2 = (-dir2_raw[0], -dir2_raw[1])

    # Oblicz kąt między kierunkami
    dot = dir1[0]*dir2[0] + dir1[1]*dir2[1]
    dot = max(-1, min(1, dot))
    angle = math.degrees(math.acos(dot))

    # Kąt 0 = ta sama linia prosta, kąt 180 = zawracanie
    # Chcemy odrzucić zakręty > 80 stopni, czyli kąt < 100 stopni
    return angle >= (180 - max_angle), angle


# Łączenie klastrów
print("Szukanie połączeń między odcinkami...")
from json_scraper import arr

connections_made = 0
connected_pairs = set()  # Aby nie łączyć tych samych dwukrotnie
new_mask = result_mask.copy()

for cluster in valid_clusters_for_connection:
    path = cluster.longest_path
    if path is None or len(path) < 20:
        continue

    # Sprawdź oba końce
    for end_idx in [0, -1]:
        endpoint = path[end_idx]

        # Oblicz kierunek na tym końcu
        sample_len = min(20, len(path)//4)
        if end_idx == 0:
            direction = get_direction_vector(path, from_end=False, sample_length=sample_len)
        else:
            direction = get_direction_vector(path, from_end=True, sample_length=sample_len)

        if direction == (0, 0):
            continue

        # Szukaj kandydatów w stożku
        candidates = find_vectors_in_cone(
            endpoint, direction, valid_clusters_for_connection, cluster,
            cone_angle=VECTOR_CONNECT_CONE_ANGLE,
            max_distance=VECTOR_CONNECT_MAX_DISTANCE
        )

        for other_cluster, other_end, dist, angle, other_end_idx in candidates:
            # Sprawdź czy już połączone
            pair_key = tuple(sorted([id(cluster), id(other_cluster)]))
            if pair_key in connected_pairs:
                continue

            # Sprawdź kąt połączenia
            angle_ok, connection_angle = check_connection_angle(
                cluster, end_idx, other_cluster, other_end_idx,
                max_angle=MAX_SHARP_ANGLE_AFTER_CONNECT
            )

            if not angle_ok:
                continue

            # Próbuj połączyć przez piksele
            connecting_pixels = connect_clusters_with_pixels(
                endpoint, other_end, new_mask, all_bands, arr, ndvi,
                NDVI_MIN_LOOSE, NDVI_MAX_LOOSE,
                min_bands_match=VECTOR_CONNECT_MIN_BANDS
            )

            if connecting_pixels is not None:
                # Dodaj piksele do maski
                for r, c in connecting_pixels:
                    new_mask[r, c] = True

                connected_pairs.add(pair_key)
                connections_made += 1
                print(f"  Połączono: depth {cluster.depth} <-> {other_cluster.depth}, "
                      f"dist={dist:.0f}, angle={connection_angle:.1f}°, "
                      f"nowe piksele={len(connecting_pixels)}")
                break  # Jeden koniec - jedno połączenie

print(f"Utworzono {connections_made} połączeń")

# Zapisz zaktualizowaną maskę
safe_rasterio_write('result_mask_connected.tif', new_mask.astype('uint8'),
                    new_mask.shape[0], new_mask.shape[1], crs, transform)


# ============ TRZECIA WEKTORYZACJA - FINALNA ============
print("\n" + "="*60)
print("TRZECIA WEKTORYZACJA - Finalna z filtrowaniem zakrętów")
print("="*60)

# Ponowna klasteryzacja na połączonej masce
labeled_final, num_final = label_with_diagonals(new_mask)
final_clusters = create_clusters(labeled_final)
print(f"Klastrów po połączeniu: {len(final_clusters)}")

# Oblicz głębokości
for cluster in final_clusters.values():
    if cluster.size() >= 50:
        cluster.get_depth()


def check_path_has_sharp_turns(path, max_angle=80, sample_step=10):
    """
    Sprawdza czy ścieżka ma ostre zakręty (> max_angle stopni).
    Zwraca (ma_ostre_zakręty, lista_kątów).
    """
    if len(path) < sample_step * 2 + 1:
        return False, []

    angles = []
    has_sharp = False

    for i in range(sample_step, len(path) - sample_step, sample_step):
        p1 = path[i - sample_step]
        p2 = path[i]
        p3 = path[i + sample_step]

        angle = calculate_angle(p1, p2, p3)
        angles.append(angle)

        # Kąt < (180 - max_angle) oznacza zakręt > max_angle stopni
        if angle < (180 - max_angle):
            has_sharp = True

    return has_sharp, angles


# Wektoryzacja finalna z filtrowaniem zakrętów
final_data = {'geometry': [], 'depth': [], 'thickness': [], 'avg_width': []}
skipped_sharp_turns = 0
skipped_highway = 0

final_clusters_list = list(final_clusters.values())
final_clusters_list.sort(key=lambda c: c.depth, reverse=True)

for cluster in final_clusters_list:
    if cluster.depth < MIN_DEPTH_FOR_VECTORIZATION:
        continue

    thickness = calculate_cluster_thickness(cluster)
    if thickness < MIN_CLUSTER_THICKNESS or thickness > MAX_CLUSTER_THICKNESS:
        continue

    # Sprawdź szerokość
    is_valid_width, avg_width, width_var, reject_reason = is_valid_railway_width(
        cluster,
        min_width=EXPECTED_TRACK_WIDTH_MIN,
        max_width=EXPECTED_TRACK_WIDTH_MAX,
        max_variance=WIDTH_VARIANCE_THRESHOLD,
        min_valid_ratio=MIN_VALID_WIDTH_RATIO
    )

    if not is_valid_width:
        # Sprawdź czy to autostrada (szeroka, stała szerokość, długa)
        if avg_width >= HIGHWAY_MIN_WIDTH and width_var <= HIGHWAY_LOW_VARIANCE and cluster.depth >= HIGHWAY_MIN_LENGTH:
            skipped_highway += 1
            print(f"  Odrzucono autostradę: depth={cluster.depth}, width={avg_width:.1f}, var={width_var:.2f}")
        continue

    # NOWY FILTR: Sprawdź ostre zakręty
    path = cluster.longest_path
    if path is not None and len(path) >= 30:
        has_sharp, angles = check_path_has_sharp_turns(path, max_angle=MAX_SHARP_ANGLE_AFTER_CONNECT)
        if has_sharp:
            min_angle = min(angles) if angles else 180
            skipped_sharp_turns += 1
            print(f"  Odrzucono (ostry zakręt): depth={cluster.depth}, min_angle={min_angle:.1f}°")
            continue

    full_path = cluster.longest_path
    steps = full_path[::10]
    geo_coords = []
    for row, col in steps:
        x, y = transform * (col, row)
        geo_coords.append((x, y))

    line = LineString(geo_coords)
    final_data['geometry'].append(line)
    final_data['depth'].append(cluster.depth)
    final_data['thickness'].append(thickness)
    final_data['avg_width'].append(avg_width)

print(f"\nStatystyki końcowe:")
print(f"  Odrzucono: {skipped_sharp_turns} z ostrymi zakrętami, {skipped_highway} autostrad")
print(f"  Zachowano: {len(final_data['geometry'])} tras kolejowych")

final_gdf = GeoDataFrame(final_data, crs=crs)

# Bezpieczny zapis końcowego GeoJSON
final_geojson_path = 'train_tracks.geojson'
try:
    final_gdf.to_file(final_geojson_path, driver='GeoJSON')
    print(f"  Zapisano: {final_geojson_path}")
except PermissionError:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    alt_path = f"train_tracks_{timestamp}.geojson"
    print(f"  Zapisuję jako: {alt_path}")
    final_gdf.to_file(alt_path, driver='GeoJSON')

print("\n" + "="*60)
print("ZAKOŃCZONO")
print("="*60)

