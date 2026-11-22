from pathlib import Path

import polyscope as ps

from assetforge.core.mesher import cylinder, save_obj

V, F = cylinder(radius=1.0, height=2.0, n=360, rings=120)

Path("out").mkdir(exist_ok=True)
save_obj("out/cylinder.obj", V, F)
print(f"{len(V)} nokta, {len(F)} üçgen -> out/cylinder.obj")

ps.init()
ps.register_surface_mesh("silindir", V, F, edge_width=1.0)
ps.show()