"""Canlı sweep demosu: elipse üç tıkla, sonra fareyi sürükleyerek parçayı "tara".

Reponun kökünden çalıştır:
    python -m scripts.live_sweep data/images/real/cini_vazo.png
Pencere kapanınca model Polyscope'ta açılır ve out/live_model.obj olarak kaydedilir.
"""

import sys
from pathlib import Path

import cv2
import polyscope as ps

from assetforge.core.camera import Camera
from assetforge.core.mesher import save_obj
from assetforge.core.sweep import sweep
from assetforge.image.silhouette import object_mask
from assetforge.ui.sweep_tool import SweepTool

if len(sys.argv) < 2:
    sys.exit("Kullanım: python -m scripts.live_sweep <fotoğraf yolu>")

img = cv2.imread(sys.argv[1])
if img is None:
    sys.exit(f"Resim okunamadı: {sys.argv[1]}")

# Büyük fotoğraflarda pencere her harekette yavaş çizilir; en uzun kenarı 900 piksele indir
k = 900 / max(img.shape[:2])
if k < 1:
    img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
h, w = img.shape[:2]

cam = Camera(scale=h / 4, cx=w / 2, cy=h / 2)     # önden bakan ortografik kamera
mask = object_mask(img)

result = SweepTool(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask, cam).run()
if result is None:
    sys.exit("Parça oluşturulmadı")

P, radii = result
V, F = sweep(P, radii, n=48)
Path("out").mkdir(exist_ok=True)
save_obj("out/live_model.obj", V, F)
print(f"out/live_model.obj kaydedildi: {len(P)} halka")

ps.init()
ps.register_surface_mesh("parça", V, F, smooth_shade=True)
ps.show()