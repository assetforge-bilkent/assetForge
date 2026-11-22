"""M4: kullanıcının fotoğraf üzerine çizdiği iki çizgiden 3D bir daire (kesit) çıkarır.

Bir bardağın ağzı fotoğrafta elips görünür. Kullanıcı:
  1. çizgi: elipsin bir ucundan öbür ucuna (uzun eksen)
  2. çizgi: elipsin merkezinden kenarına, uzun eksene dik (kısa eksen)
Kısa / uzun oranı, dairenin kameraya ne kadar eğik durduğunu söyler:
  oran = 1  -> daireye tam karşıdan bakıyoruz
  oran = 0  -> daireye tam yandan bakıyoruz (düz çizgi görünür)
"""

from dataclasses import dataclass

import numpy as np

from assetforge.core.camera import unproject


@dataclass
class Circle3D:
    center: np.ndarray   # (3,) dairenin merkezi
    normal: np.ndarray   # (3,) dairenin düzlemine dik birim vektör (parçanın ekseni buradan başlar)
    radius: float


def image_dir_to_world(d, cam):
    """Resimdeki bir yönü (u sağa, v aşağı) dünyadaki karşılığına çevirir."""
    return d[0] * cam.R[0] - d[1] * cam.R[1]


def circle_from_strokes(major_a, major_b, minor_end, cam, flip=False):
    """İki çizgiden 3D daire.

    major_a, major_b: 1. çizginin uçları (piksel)
    minor_end:        2. çizginin bittiği nokta (piksel); başlangıcı elipsin merkezi kabul edilir
    flip:             aynı elipsi veren iki olası daireden diğerini seçer
                      (varsayılan: normal resimde yukarı doğru eğik, yani nesneye üstten bakıyoruz)
    """
    a, b, m = (np.asarray(p, dtype=float) for p in (major_a, major_b, minor_end))
    center_px = (a + b) / 2
    r_px = np.linalg.norm(b - a) / 2
    if r_px < 1:
        raise ValueError("Uzun eksen çizgisi çok kısa")

    d1 = (b - a) / (2 * r_px)                    # uzun eksenin resimdeki yönü
    d2 = np.array([-d1[1], d1[0]])               # ona dik yön (kısa eksen)
    minor_px = abs(np.dot(m - center_px, d2))    # sadece dik bileşeni say
    k = min(minor_px / r_px, 1.0)                # kısa / uzun oranı

    if d2[1] > 0:                                # d2 resimde yukarıyı göstersin (v azalan)
        d2 = -d2
    if flip:
        d2 = -d2

    in_image = image_dir_to_world(d2, cam)       # normalin resim düzlemindeki kısmı
    toward_camera = cam.R[2]
    normal = np.sqrt(1 - k * k) * in_image + k * toward_camera
    normal /= np.linalg.norm(normal)

    center = unproject(center_px[None], 0.0, cam)[0]
    return Circle3D(center=center, normal=normal, radius=r_px / cam.scale)


def circle_points(circle, n=64):
    """Dairenin üzerinde n nokta (önizleme çizmek için)."""
    nrm = circle.normal
    helper = np.array([0.0, 0.0, 1.0]) if abs(nrm[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(nrm, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nrm, e1)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return circle.center + circle.radius * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))