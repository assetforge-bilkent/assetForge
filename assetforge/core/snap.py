"""M5: kullanıcının üçüncü çizgisi boyunca halkaları fotoğraftaki nesnenin kenarlarına yapıştırır.

Girdi: M4'ten gelen başlangıç dairesi, nesne maskesi ve çizginin bittiği nokta.
Çıktı: sweep'e verilecek eksen noktaları ve yarıçaplar.

Bu sürümde eksen düz (başlangıç dairesinin normali boyunca); yarıçap ve eksenin
yana kayması her halkada fotoğraftan ölçülür. Eğri eksen ileride eklenecek.
"""

import numpy as np

from assetforge.core.camera import project
from assetforge.core.cross_section import image_dir_to_world

def walk_to_edge(mask, start, direction, max_dist=None, step=0.5):
    """Maskenin içindeki bir noktadan verilen yönde yürür, maskeden çıkana kadar kaç piksel gittiğini döndürür.

    Başlangıç noktası maskenin dışındaysa None döndürür.
    Hız için adımları tek tek denemek yerine, yol üzerindeki bütün noktalara NumPy ile bir kerede bakar.
    """
    h, w = mask.shape
    p = np.asarray(start, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    if max_dist is None:
        max_dist = np.hypot(h, w)                    # resmin köşegeni: daha uzağa gidilemez

    dist = np.arange(0, max_dist, step)              # 0, 0.5, 1.0, ... piksel
    u = np.round(p[0] + dist * d[0]).astype(int)
    v = np.round(p[1] + dist * d[1]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    inside[inside] = mask[v[inside], u[inside]] > 0

    if not inside[0]:
        return None
    outside = np.flatnonzero(~inside)
    if len(outside) == 0:
        return float(max_dist)
    return float(dist[outside[0]] - step / 2)

    def inside(q):
        u, v = int(round(q[0])), int(round(q[1]))
        return 0 <= u < w and 0 <= v < h and mask[v, u] > 0

    if not inside(p):
        return None
    for i in range(1, max_steps):
        if not inside(p + i * step * d):
            return (i - 0.5) * step
    return max_steps * step


def smooth(values, k=5):
    """Kayan ortalama: k komşunun ortalaması. Uçlarda pencere kısalır."""
    values = np.asarray(values, dtype=float)
    if k <= 1 or len(values) < 3:
        return values
    half = k // 2
    return np.array([values[max(0, i - half):i + half + 1].mean() for i in range(len(values))])


def snap_sweep(mask, circle, axis_end_px, cam, n_rings=40, smoothing=5):
    """Başlangıç dairesinden, kullanıcının çizgisinin bittiği noktaya kadar halkaları kenarlara oturtur.

    mask:        nesne maskesi (nesne 255)
    circle:      M4'ten gelen Circle3D (parçanın bir ucundaki kesit)
    axis_end_px: üçüncü çizginin bittiği piksel (parçanın öbür ucu)
    Döndürür: (P, radii) -> doğrudan sweep(P, radii) ile modele çevrilebilir.
    """
    n = np.asarray(circle.normal, dtype=float)
    center_px = project(circle.center[None], cam)[0]

    # Eksenin resimdeki yönü: normalin resim düzlemine düşen kısmı
    t2d = np.array([n @ cam.R[0], -(n @ cam.R[1])])
    shrink = np.linalg.norm(t2d)              # 1 birim eksen boyu resimde kaç birim görünür
    if shrink < 0.05:
        raise ValueError("Eksen neredeyse kameraya doğru bakıyor; bu açıdan uzunluk ölçülemez")
    t2d /= shrink

    # Çizgi ters yöndeyse ekseni çevir (parça dairenin öbür tarafına uzanıyor)
    along_px = np.dot(np.asarray(axis_end_px, dtype=float) - center_px, t2d)
    if along_px < 0:
        n, t2d, along_px = -n, -t2d, -along_px
    length = along_px / (cam.scale * shrink)  # eksenin 3D uzunluğu

    q2d = np.array([-t2d[1], t2d[0]])         # resimde eksene dik yön
    q3d = image_dir_to_world(q2d, cam)        # onun dünyadaki karşılığı

    s = np.linspace(0, length, n_rings)
    base = circle.center + np.outer(s, n)     # düz eksen üzerindeki noktalar
    base_px = project(base, cam)

    radii, shifts, keep = [], [], []
    for i, p in enumerate(base_px):
        left = walk_to_edge(mask, p, -q2d)
        right = walk_to_edge(mask, p, q2d)
        if left is None or right is None:     # bu halka nesnenin dışında kaldı
            continue
        radii.append((left + right) / 2 / cam.scale)
        shifts.append((right - left) / 2 / cam.scale)
        keep.append(i)

    if len(keep) < 2:
        raise ValueError("Çizgi boyunca nesne bulunamadı; başlangıç dairesi maskenin içinde mi?")

    radii = smooth(radii, smoothing)
    shifts = smooth(shifts, smoothing)
    P = base[keep] + np.outer(shifts, q3d)
    return P, radii