"""Sweep: bir çemberi (kesiti) bir eksen boyunca kaydırarak 3D parça üretir."""

import numpy as np

from assetforge.core.frames import frames
from assetforge.core.mesher import cap_faces, grid_faces


def sweep(P, radii, n=36, caps=True):
    """Eksen P boyunca, her noktada verilen yarıçapta halkalar dizer. (V, F) döndürür.

    P:     (k, 3) eksen noktaları
    radii: tek bir sayı (sabit kalınlık) ya da her eksen noktası için bir yarıçap
    n:     her halkadaki nokta sayısı
    """
    P = np.asarray(P, dtype=float)
    radii = np.broadcast_to(np.asarray(radii, dtype=float), (len(P),))
    T, N, B = frames(P)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)

    # 1) Noktalar: her eksen noktasında, N-B düzlemine bir halka
    points = []
    for i in range(len(P)):
        for c, s in zip(cos, sin):
            points.append(P[i] + radii[i] * (c * N[i] + s * B[i]))
    V = np.array(points)

    # 2) Yan yüz: silindirdeki ile aynı fonksiyon
    F = grid_faces(len(P), n)

    # 3) Kapaklar: merkezleri eksenin ilk ve son noktası
    if caps:
        start_center, end_center = len(V), len(V) + 1
        V = np.vstack([V, P[0], P[-1]])
        start = cap_faces(0, n, start_center, flip=True)
        end = cap_faces((len(P) - 1) * n, n, end_center, flip=False)
        F = np.vstack([F, start, end])

    return V, F


def arc_axis(radius=2.0, angle_deg=90, steps=30):
    """Örnek eksen: xz düzleminde bir çember yayı (dirsek boru)."""
    t = np.linspace(0, np.radians(angle_deg), steps)
    return np.c_[radius * (1 - np.cos(t)), np.zeros(steps), radius * np.sin(t)]