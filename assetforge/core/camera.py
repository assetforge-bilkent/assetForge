"""Kamera: 3D noktaların fotoğrafta hangi piksele düştüğünü hesaplar.

Ortografik kamera kullanıyoruz: uzaktaki nesneler küçülmez (perspektif yok).
Dünya koordinatları: x sağa, y ileri (kameradan uzağa), z yukarı.
Resim koordinatları: u sağa, v aşağı (piksel).
"""

from dataclasses import dataclass, field

import numpy as np


def view_rotation(elevation_deg=0.0):
    """Kameranın bakış yönünü veren 3x3 döndürme matrisi.

    elevation_deg = 0  : nesneye tam önden bakılır (z yukarısı resimde yukarı)
    elevation_deg = 30 : kamera 30 derece yukarıdan, aşağıya doğru bakar
    elevation_deg = 90 : tam tepeden bakılır

    Satırlar sırasıyla dünyada "resmin sağı", "resmin yukarısı" ve "kameraya doğru" yönleridir.
    """
    e = np.radians(elevation_deg)
    s, c = np.sin(e), np.cos(e)
    return np.array([
        [1.0, 0.0, 0.0],   # resmin sağı  = dünyanın x'i
        [0.0, s,   c],     # resmin yukarısı
        [0.0, -c,  s],     # kameraya doğru (derinlik)
    ])


@dataclass
class Camera:
    """Ortografik kamera.

    scale: 1 dünya birimi kaç piksel
    cx, cy: dünyanın orijininin (0, 0, 0) resimdeki piksel konumu
    R: bakış yönü (view_rotation ile üretilir)
    """
    scale: float
    cx: float
    cy: float
    R: np.ndarray = field(default_factory=lambda: view_rotation(0.0))


def project(X, cam):
    """3D noktaları (k, 3) piksel koordinatlarına (k, 2) çevirir."""
    X = np.asarray(X, dtype=float)
    Xc = X @ cam.R.T                       # kameranın gözünden koordinatlar
    u = cam.cx + cam.scale * Xc[:, 0]
    v = cam.cy - cam.scale * Xc[:, 1]      # resimde v aşağı doğru artar
    return np.c_[u, v]


def depth(X, cam):
    """Her noktanın kameraya ne kadar yakın olduğu (büyük = daha yakın)."""
    return np.asarray(X, dtype=float) @ cam.R[2]


def unproject(uv, d, cam):
    """project'in tersi: piksel (k, 2) ve derinlik (k,) verilince 3D noktaları bulur."""
    uv = np.asarray(uv, dtype=float)
    d = np.broadcast_to(np.asarray(d, dtype=float), (len(uv),))
    Xc = np.c_[(uv[:, 0] - cam.cx) / cam.scale, (cam.cy - uv[:, 1]) / cam.scale, d]
    return Xc @ cam.R                      # R'nin tersi, devriğidir


def mesh_edges(F):
    """Üçgen listesinden tekrarsız kenar listesi (tel kafes çizmek için)."""
    F = np.asarray(F)
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)
