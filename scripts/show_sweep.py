"""M1 demo: eğri eksen boyunca sweep. Bir dirsek boru ve bir vazo gösterir.

Reponun kökünden çalıştır:
    python -m scripts.show_sweep
"""

import numpy as np
import polyscope as ps

from assetforge.core.sweep import arc_axis, sweep

# 1) Dirsek boru: çeyrek çember şeklinde eksen, sabit kalınlık
pipe_axis = arc_axis(radius=2.0, angle_deg=90, steps=30)
pipe_V, pipe_F = sweep(pipe_axis, radii=0.3)

# 2) Vazo: dümdüz eksen, yüksekliğe göre değişen yarıçap
k = 25
z = np.linspace(0, 3, k)
vase_axis = np.c_[np.full(k, 4.0), np.zeros(k), z]        # x = 4'e kaydırdık, yan yana dursunlar
vase_radii = 0.5 + 0.35 * np.sin(np.linspace(0, 1.5 * np.pi, k))
vase_V, vase_F = sweep(vase_axis, radii=vase_radii)

ps.init()
ps.register_surface_mesh("dirsek boru", pipe_V, pipe_F, edge_width=1.0)
ps.register_surface_mesh("vazo", vase_V, vase_F, edge_width=1.0)

# Eksenleri de çizgi olarak göster: halkaların neye göre dizildiğini görmek için
for name, axis in [("boru ekseni", pipe_axis), ("vazo ekseni", vase_axis)]:
    edges = np.c_[np.arange(len(axis) - 1), np.arange(1, len(axis))]
    ps.register_curve_network(name, axis, edges, radius=0.005)

ps.show()