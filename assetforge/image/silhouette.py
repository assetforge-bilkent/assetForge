"""Siluetten model: dik duran, simetrik bir nesnenin (vazo, şişe, bardak) fotoğrafından
eksen ve yarıçapları çıkarır. Sonuç doğrudan sweep'e verilir.

Varsayımlar (bu ilk sürüm için):
- Fotoğraf nesneye tam önden çekilmiş.
- Nesne dik duruyor ve dönel simetrik (her yüksekliğin kesiti bir daire).
- Arka plan sade (tek renk duvar, masa örtüsü).
"""

import cv2
import numpy as np


def object_mask(img, method="auto", rect=None):
    """Nesneyi arka plandan ayırır. Nesne 255, arka plan 0 olan bir maske döndürür.

    method="auto":    önce arka plan rengine göre ayırmayı dener (hızlı, sade arka planda
                      çok iyi); sonuç anlamsızsa GrabCut'a geçer.
    method="color":   sadece arka plan rengine göre ayır.
    method="grabcut": sadece GrabCut kullan (rect verilebilir).
    """
    if method in ("auto", "color"):
        mask = color_mask(img)
        area = (mask > 0).mean()
        if method == "color" or 0.02 < area < 0.9:
            return mask
    return grabcut_mask(img, rect)


def color_mask(img):
    """Arka planın rengini resmin kenarlarından öğrenir, ondan farklı pikselleri nesne sayar.

    Lab renk uzayını kullanıyoruz: L parlaklık, a ve b renk tonu. Arka plan renkliyse
    (kırmızı, mavi...) sadece renk tonuna bakmak gölgeleri ve parlaklık farklarını yok sayar.
    Arka plan griyse (beyaz duvar) parlaklık da hesaba katılır.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = 10                                                     # kenar şeridinin kalınlığı
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)

    bg_chroma = np.hypot(bg[1] - 128, bg[2] - 128)            # arka plan ne kadar renkli
    channels = slice(1, 3) if bg_chroma > 15 else slice(0, 3)
    dist = np.linalg.norm(lab[:, :, channels] - bg[channels], axis=2)

    dist8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return clean_mask(mask)


def grabcut_mask(img, rect=None, max_side=800):
    """GrabCut ile nesneyi ayırır. rect: (x, y, w, h) nesnenin içinde olduğu kutu.

    Verilmezse resmin kenarlarından %5 içeride bir kutu kullanılır.
    GrabCut yavaş olduğu için büyük resimler önce küçültülür, maske sonra büyütülür.
    """
    h, w = img.shape[:2]
    k = min(1.0, max_side / max(h, w))
    small = cv2.resize(img, (round(w * k), round(h * k))) if k < 1 else img
    sh, sw = small.shape[:2]

    if rect is None:
        mx, my = round(sw * 0.05), round(sh * 0.05)
        rect = (mx, my, sw - 2 * mx, sh - 2 * my)
    else:
        rect = tuple(round(v * k) for v in rect)

    gc = np.zeros((sh, sw), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    if k < 1:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return clean_mask(mask)


def clean_mask(mask):
    """Maskeyi temizler: küçük delikleri kapatır, kırıntıları atar, en büyük parçayı tutar."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
    if count <= 1:
        return np.zeros_like(mask)
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])   # 0 numara arka plan
    m = np.where(labels == biggest, 255, 0).astype(np.uint8)

    # İçerideki delikleri doldur: dış sınırı bulup içini tamamen boya
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def row_profile(mask):
    """Maskenin her satırında nesnenin sol ve sağ kenarını bulur.

    Döndürür: rows (satır numaraları, yukarıdan aşağı), centers (orta nokta, piksel),
    radii (yarı genişlik, piksel). Nesnenin hiç olmadığı satırlar atlanır.
    """
    rows, centers, radii = [], [], []
    for v in range(mask.shape[0]):
        cols = np.flatnonzero(mask[v])
        if len(cols) == 0:
            continue
        left, right = cols[0], cols[-1]
        rows.append(v)
        centers.append((left + right) / 2)
        radii.append((right - left + 1) / 2)
    return np.array(rows), np.array(centers), np.array(radii)


def axis_and_radii_from_mask(mask, n_rings=40, height=1.0, smooth=5):
    """Maskeden sweep için dik bir eksen ve her halkanın yarıçapını çıkarır.

    n_rings: eksen boyunca kaç halka olacağı
    height:  modelin yüksekliği (dünya biriminde); fotoğraftan gerçek boy bilinemez
    smooth:  yarıçapları kaç satır üzerinden yumuşatacağı (titremeyi önler)
    Döndürür: (P, radii), P = (n_rings, 3) eksen noktaları, z yukarı.
    """
    rows, _, radii_px = row_profile(mask)
    if len(rows) < 2:
        raise ValueError("Maskede nesne bulunamadı")

    if smooth > 1 and len(radii_px) >= smooth:
        radii_px = np.convolve(radii_px, np.ones(smooth) / smooth, mode="same")

    top, bottom = rows[0], rows[-1]
    px_per_unit = (bottom - top) / height

    sample_rows = np.linspace(bottom, top, n_rings)          # aşağıdan yukarı
    sample_radii = np.interp(sample_rows, rows, radii_px) / px_per_unit
    z = (bottom - sample_rows) / px_per_unit

    P = np.c_[np.zeros(n_rings), np.zeros(n_rings), z]
    return P, sample_radii
