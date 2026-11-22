"""Sentetik bir vazo "fotoğrafı" üretir: ölçülerini bildiğimiz için doğruluk testinde kullanılır.

Reponun kökünden çalıştır:
    python -m scripts.make_test_vase
Çıktı: data/images/synthetic/vase.png
"""

from pathlib import Path

import cv2
import numpy as np

h, w, px_per_unit = 800, 600, 600
z = np.linspace(0, 1, 120)
r = 0.15 + 0.08 * np.sin(z * 1.5 * np.pi)          # gerçek yarıçaplar (yükseklik = 1)
cx, bottom = w / 2, h - 100

img = np.full((h, w, 3), 230, np.uint8)            # açık gri duvar

# Vazoyu yatay şeritler halinde çiz; ortası açık, kenarları koyu (basit gölgelendirme)
for zi, ri in zip(z, r):
    v = int(round(bottom - zi * px_per_unit))
    half = ri * px_per_unit
    for u in range(int(cx - half), int(cx + half) + 1):
        t = (u - cx) / half                          # -1 (sol kenar) ... 1 (sağ kenar)
        shade = 0.55 + 0.45 * np.sqrt(max(0.0, 1 - t * t))
        img[v - 3:v + 3, u] = (np.array([60, 110, 190]) * shade).astype(np.uint8)

noise = np.random.default_rng(0).normal(0, 4, img.shape)
img = np.clip(img + noise, 0, 255).astype(np.uint8)

out = Path("data/images/synthetic")
out.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out / "vase.png"), img)
print("data/images/synthetic/vase.png kaydedildi (gerçek boy 1, en geniş yarıçap ~0.23)")
