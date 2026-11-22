"""Siluetten model: bir vazo/şişe/bardak fotoğrafından 3D model üretir.

Reponun kökünden çalıştır:
    python -m scripts.photo_to_model data/images/synthetic/vase.png
Çıktılar:
    out/mask_overlay.png  bulunan siluet, fotoğrafın üstünde yeşil çizgiyle
    out/model.obj         3D model (Blender'da açılabilir)
Sonra model Polyscope'ta gösterilir.
"""

import sys
from pathlib import Path

import cv2
import polyscope as ps

from assetforge.core.mesher import save_obj
from assetforge.core.sweep import sweep
from assetforge.image.silhouette import axis_and_radii_from_mask, object_mask

if len(sys.argv) < 2:
    sys.exit("Kullanım: python -m scripts.photo_to_model <fotoğraf yolu>")

img = cv2.imread(sys.argv[1])
if img is None:
    sys.exit(f"Resim okunamadı: {sys.argv[1]}")

# 1) Nesneyi arka plandan ayır
mask = object_mask(img)

# 2) Kontrol için siluetin sınırını fotoğrafın üstüne çiz ve kaydet
Path("out").mkdir(exist_ok=True)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
overlay = img.copy()
cv2.drawContours(overlay, contours, -1, (0, 200, 0), 2)
cv2.imwrite("out/mask_overlay.png", overlay)
print("out/mask_overlay.png kaydedildi: yeşil çizgi nesneyi doğru sarıyor mu, bak")

# 3) Siluetten eksen ve yarıçapları çıkar, sweep ile 3D modele çevir
P, radii = axis_and_radii_from_mask(mask, n_rings=60, height=1.0)
V, F = sweep(P, radii, n=48)
save_obj("out/model.obj", V, F)
print(f"out/model.obj kaydedildi: {len(V)} nokta, {len(F)} üçgen")

# 4) Göster
ps.init()
ps.register_surface_mesh("model", V, F, smooth_shade=True)
ps.show()
