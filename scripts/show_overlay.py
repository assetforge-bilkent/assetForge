"""M2 demo: 3D parçaların tel kafesini bir resmin üstüne çizer.

Reponun kökünden çalıştır:
    python -m scripts.show_overlay                     # beyaz zemin
    python -m scripts.show_overlay data/images/x.jpg   # kendi fotoğrafın
Sonuç out/overlay.png olarak da kaydedilir. Pencereyi kapatmak için bir tuşa bas.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from assetforge.core.camera import Camera, mesh_edges, project, view_rotation
from assetforge.core.sweep import arc_axis, sweep

# Arka plan: fotoğraf verilmişse onu, verilmemişse beyaz bir zemin kullan
if len(sys.argv) > 1:
    image = cv2.imread(sys.argv[1])
    if image is None:
        sys.exit(f"Resim okunamadı: {sys.argv[1]}")
else:
    image = np.full((600, 800, 3), 255, dtype=np.uint8)

h, w = image.shape[:2]
cam = Camera(scale=min(w, h) / 6, cx=w / 2, cy=h * 0.8, R=view_rotation(elevation_deg=20))

# Gösterilecek parçalar: bir vazo ve yanında bir dirsek boru
k = 25
vase_axis = np.c_[np.full(k, -1.2), np.zeros(k), np.linspace(0, 3, k)]
vase = sweep(vase_axis, 0.5 + 0.35 * np.sin(np.linspace(0, 1.5 * np.pi, k)), n=24)
pipe = sweep(arc_axis(radius=1.5, angle_deg=90, steps=20) + [0.6, 0, 0], 0.25, n=16)

for (V, F), color in [(vase, (200, 90, 30)), (pipe, (40, 40, 220))]:   # renkler BGR
    uv = project(V, cam).round().astype(int)
    for a, b in mesh_edges(F):
        cv2.line(image, tuple(uv[a]), tuple(uv[b]), color, 1, cv2.LINE_AA)

Path("out").mkdir(exist_ok=True)
cv2.imwrite("out/overlay.png", image)
print("out/overlay.png kaydedildi")

cv2.imshow("AssetForge - overlay", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
