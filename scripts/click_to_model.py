"""M4 + M5 demo: fotoğrafa 4 kez tıklayarak bir parçanın 3D modelini üretir.

Reponun kökünden çalıştır:
    python -m scripts.click_to_model data/images/synthetic/vase.png

Tıklama sırası (açılan pencerede):
  1. Parçanın bir ucundaki elipsin SOL ucu
  2. Aynı elipsin SAĞ ucu                     -> uzun eksen (1. çizgi)
  3. Elipsin ön kenarı (merkezin altında)      -> kısa eksen (2. çizgi)
  4. Parçanın öbür ucu                         -> eksen (3. çizgi)
Yanlış tıklarsan sağ tık son tıklamayı geri alır.

Çıktılar: out/click_overlay.png (modelin fotoğraf üstündeki tel kafesi), out/click_model.obj
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import polyscope as ps

from assetforge.core.camera import Camera, mesh_edges, project
from assetforge.core.cross_section import circle_from_strokes
from assetforge.core.mesher import save_obj
from assetforge.core.snap import snap_sweep
from assetforge.core.sweep import sweep
from assetforge.image.silhouette import object_mask

if len(sys.argv) < 2:
    sys.exit("Kullanım: python -m scripts.click_to_model <fotoğraf yolu>")

img = cv2.imread(sys.argv[1])
if img is None:
    sys.exit(f"Resim okunamadı: {sys.argv[1]}")
h, w = img.shape[:2]

# Fotoğrafın gerçek ölçeğini bilmiyoruz: resim yüksekliği 4 birim olsun diye kabul ediyoruz.
# Önden bakan ortografik kamera; eğiklik bilgisini kullanıcının çizdiği elips veriyor.
cam = Camera(scale=h / 4, cx=w / 2, cy=h / 2)

mask = object_mask(img)

# 1) Kullanıcıdan 4 tıklama al
plt.figure(figsize=(7, 7 * h / w))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("4 tıklama: elipsin solu, sağı, ön kenarı, parçanın öbür ucu")
plt.axis("off")
clicks = plt.ginput(4, timeout=0)
plt.close()
if len(clicks) < 4:
    sys.exit("4 tıklama gerekli")
left, right, front, end = (np.array(c) for c in clicks)

# 2) M4: iki çizgiden başlangıç dairesi
circle = circle_from_strokes(left, right, front, cam)

# 3) M5: üçüncü çizgi boyunca halkaları kenarlara yapıştır
P, radii = snap_sweep(mask, circle, end, cam, n_rings=50)
V, F = sweep(P, radii, n=48)

# 4) Kontrol: modelin tel kafesini fotoğrafın üstüne çiz
overlay = img.copy()
uv = project(V, cam).round().astype(int)
for a, b in mesh_edges(F):
    cv2.line(overlay, tuple(uv[a]), tuple(uv[b]), (0, 200, 0), 1, cv2.LINE_AA)
Path("out").mkdir(exist_ok=True)
cv2.imwrite("out/click_overlay.png", overlay)
save_obj("out/click_model.obj", V, F)
print("out/click_overlay.png ve out/click_model.obj kaydedildi")

# 5) 3D göster
ps.init()
ps.register_surface_mesh("parça", V, F, smooth_shade=True)
ps.show()