"""Canlı sweep aracı: fotoğrafın üstünde elipsi gösterip fareyi sürükleyerek parça oluşturma.

Kullanım (pencerede):
  Sol tık x3   : elipsin sol ucu, sağ ucu, ön kenarı      (M4: başlangıç dairesi)
  Sürükle      : fareyi basılı tutup parçanın ekseni boyunca çek, halkalar canlı yapışır (M5)
  Bırak        : parça tamamlanır, pencere kapanınca 3D model gösterilir
  Sağ tık      : son tıklamayı geri al
  f            : elipsin öbür çözümünü dene (alttan/üstten bakış)
  r            : baştan başla
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from assetforge.core.camera import project
from assetforge.core.cross_section import Circle3D, circle_from_strokes, circle_points
from assetforge.core.snap import snap_sweep

HELP = "Sol tık x3: elipsin solu, sağı, ön kenarı  |  sağ tık: geri al  |  f: çevir  |  r: baştan"


class SweepTool:
    def __init__(self, image_rgb, mask, cam):
        self.image, self.mask, self.cam = image_rgb, mask, cam
        self.clicks = []          # elips için tıklanan noktalar
        self.flip = False
        self.circle = None        # M4 sonucu
        self.dragging = False
        self.result = None        # (P, radii): bırakıldığında son hal

        self.fig, self.ax = plt.subplots(figsize=(8, 8 * image_rgb.shape[0] / image_rgb.shape[1]))
        self.ax.imshow(image_rgb)
        self.ax.set_axis_off()
        self.set_title(HELP, fontsize=9)

        # Çizim katmanları: tıklama noktaları, iki çizgi, başlangıç elipsi, halkalar, eksen
        (self.dots,) = self.ax.plot([], [], "o", color="yellow", ms=5)
        (self.stroke1,) = self.ax.plot([], [], "-", color="yellow", lw=1.5)
        (self.stroke2,) = self.ax.plot([], [], "-", color="orange", lw=1.5)
        (self.start_ellipse,) = self.ax.plot([], [], "-", color="cyan", lw=2)
        (self.axis_line,) = self.ax.plot([], [], "--", color="white", lw=1)
        self.rings = LineCollection([], colors="lime", linewidths=1)
        self.ax.add_collection(self.rings)

        # Hız için "blitting": fotoğraf bir kez çizilip saklanır, her harekette sadece
        # çizgiler onun üstüne yeniden çizilir (bütün pencereyi baştan çizmek çok yavaş)
        self.artists = [self.dots, self.stroke1, self.stroke2, self.start_ellipse,
                        self.axis_line, self.rings]
        for a in self.artists:
            a.set_animated(True)
        self.background = None

        c = self.fig.canvas
        c.mpl_connect("draw_event", self.on_draw)
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("motion_notify_event", self.on_move)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("key_press_event", self.on_key)

    # ---------- olaylar ----------

    def on_draw(self, event):
        """Pencere tamamen çizildiğinde (ilk açılış, yeniden boyutlandırma) arka planı sakla."""
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.blit()

    def on_press(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        p = np.array([event.xdata, event.ydata])
        if event.button == 3:                         # sağ tık: geri al
            self.undo()
        elif self.circle is None:                     # elips aşaması
            self.clicks.append(p)
            if len(self.clicks) == 3:
                self.fit_circle()
        else:                                         # sürükleme başlıyor
            self.dragging = True
            self.update_sweep(p, n_rings=None)
        self.redraw()

    def on_move(self, event):
        if self.dragging and event.inaxes is self.ax and event.xdata is not None:
            self.update_sweep(np.array([event.xdata, event.ydata]), n_rings=None)
            self.redraw()

    def on_release(self, event):
        if not self.dragging:
            return
        self.dragging = False
        if event.xdata is not None:
            self.update_sweep(np.array([event.xdata, event.ydata]), n_rings=50)
        if self.result is not None:
            self.set_title("Parça hazır: pencereyi kapat, 3D model açılacak  |  r: baştan", fontsize=9)
        self.redraw()

    def on_key(self, event):
        if event.key == "r":
            self.clicks, self.circle, self.result = [], None, None
            self.set_title(HELP, fontsize=9)
        elif event.key == "f" and len(self.clicks) == 3:
            self.flip = not self.flip
            self.fit_circle()
            self.result = None
        self.redraw()

    def set_title(self, text, fontsize=9):
        """Başlık değişince bütün pencere yeniden çizilmeli (blitting sadece çizgileri günceller)."""
        self.ax.set_title(text, fontsize=fontsize)
        self.fig.canvas.draw_idle()

    # ---------- hesap ----------

    def undo(self):
        if self.result is not None or self.circle is not None:
            self.result, self.circle = None, None
            self.clicks = self.clicks[:2]
            self.set_title(HELP, fontsize=9)
        elif self.clicks:
            self.clicks.pop()

    def fit_circle(self):
        a, b, m = self.clicks
        try:
            self.circle = circle_from_strokes(a, b, m, self.cam, flip=self.flip)
            self.set_title("Şimdi fareyi basılı tutup parçanın ekseni boyunca sürükle", fontsize=9)
        except ValueError as err:
            self.clicks.pop()
            self.set_title(f"{err}. Tekrar tıkla.", fontsize=9)

    def update_sweep(self, mouse_px, n_rings):
        center_px = project(self.circle.center[None], self.cam)[0]
        if n_rings is None:                            # sürüklerken: uzunlukla orantılı, az halka (hızlı)
            n_rings = int(np.clip(np.linalg.norm(mouse_px - center_px) / 12, 2, 30))
        try:
            self.result = snap_sweep(self.mask, self.circle, mouse_px, self.cam, n_rings=n_rings)
        except ValueError:
            pass                                       # fare nesnenin dışına çıktıysa son hali koru

    # ---------- çizim ----------

    def redraw(self):
        pts = np.array(self.clicks) if self.clicks else np.empty((0, 2))
        self.dots.set_data(pts[:, 0], pts[:, 1])
        self.stroke1.set_data(pts[:2, 0], pts[:2, 1]) if len(pts) >= 2 else self.stroke1.set_data([], [])
        if len(pts) == 3:
            mid = (pts[0] + pts[1]) / 2
            self.stroke2.set_data([mid[0], pts[2][0]], [mid[1], pts[2][1]])
        else:
            self.stroke2.set_data([], [])

        if self.circle is not None:
            e = project(circle_points(self.circle, 64), self.cam)
            e = np.vstack([e, e[:1]])
            self.start_ellipse.set_data(e[:, 0], e[:, 1])
        else:
            self.start_ellipse.set_data([], [])

        if self.result is not None:
            P, radii = self.result
            d = P[-1] - P[0]
            n = self.circle.normal if np.dot(d, self.circle.normal) >= 0 else -self.circle.normal
            segs = []
            for p, r in zip(P, radii):
                ring = project(circle_points(Circle3D(p, n, r), 40), self.cam)
                segs.append(np.vstack([ring, ring[:1]]))
            self.rings.set_segments(segs)
            axis = project(P[[0, -1]], self.cam)
            self.axis_line.set_data(axis[:, 0], axis[:, 1])
        else:
            self.rings.set_segments([])
            self.axis_line.set_data([], [])

        self.blit()

    def blit(self):
        canvas = self.fig.canvas
        if self.background is None:
            canvas.draw_idle()                         # henüz arka plan yok: normal çizim
            return
        canvas.restore_region(self.background)
        for a in self.artists:
            self.ax.draw_artist(a)
        canvas.blit(self.ax.bbox)

    def run(self):
        """Pencereyi açar; kapanınca son parçayı (P, radii) ya da None döndürür."""
        plt.show()
        return self.result