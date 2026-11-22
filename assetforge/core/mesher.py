import numpy as np


def grid_faces(n_rings, n_per_ring):
    faces = []
    for i in range(n_rings - 1):
        for j in range(n_per_ring):
            j_next = (j + 1) % n_per_ring
            p = i * n_per_ring + j
            q = i * n_per_ring + j_next
            s = (i + 1) * n_per_ring + j
            u = (i + 1) * n_per_ring + j_next
            faces.append([p, q, u])
            faces.append([p, u, s])
    return np.array(faces)

def cap_faces(ring_start, n_per_ring, center, flip):
    """Bir halkayı ortadaki bir noktaya bağlayarak kapatır (pizza dilimleri gibi).

    flip=False: üst kapak (yukarı bakar), flip=True: alt kapak (aşağı bakar).
    """
    faces = []
    for j in range(n_per_ring):
        a = ring_start + j
        b = ring_start + (j + 1) % n_per_ring
        if flip:
            faces.append([center, b, a])
        else:
            faces.append([center, a, b])
    return np.array(faces)


def cylinder(radius=1.0, height=2.0, n=36, rings=12, caps=True):
    """z ekseni boyunca duran, tabanı xy düzleminde olan silindir. (V, F) döndürür."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    heights = np.linspace(0, height, rings)

    # 1) Noktalar: halka halka, her halkada n nokta
    points = []
    for z in heights:
        for a in angles:
            x = radius * np.cos(a)
            y = radius * np.sin(a)
            points.append([x, y, z])
    V = np.array(points)

    # 2) Yan yüzün üçgenleri
    F = grid_faces(rings, n)

    # 3) Kapaklar: alt ve üst merkez noktalarını ekle, halkaları onlara bağla
    if caps:
        bottom_center = len(V)        # yeni eklenecek noktanın numarası
        top_center = len(V) + 1
        V = np.vstack([V, [0, 0, 0], [0, 0, height]])

        bottom = cap_faces(0, n, bottom_center, flip=True)
        top = cap_faces((rings - 1) * n, n, top_center, flip=False)
        F = np.vstack([F, bottom, top])

    return V, F


if __name__ == "__main__":
    V, F = cylinder()
    print(V.shape, F.shape)
    print(V[:4])

def save_obj(path, V, F):
    """Mesh'i OBJ dosyası olarak kaydeder (OBJ'de numaralar 1'den başlar)."""
    with open(path, "w") as f:
        for x, y, z in V:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in F:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")