import cv2
import numpy as np
import pytest

from assetforge.core.camera import Camera, project, view_rotation
from assetforge.core.cross_section import Circle3D, circle_from_strokes
from assetforge.core.snap import smooth, snap_sweep, walk_to_edge
from assetforge.core.sweep import sweep


def render_mask(V, F, cam, size=(600, 800)):
    """Bir mesh'in fotoğraftaki siluetini çizer (her üçgeni doldurarak)."""
    h, w = size
    mask = np.zeros((h, w), np.uint8)
    uv = np.round(project(V, cam) * 4).astype(np.int32)        # 4 kat hassas çizim
    for tri in F:
        cv2.fillPoly(mask, [uv[tri]], 255, lineType=cv2.LINE_8, shift=2)
    return mask


def make_scene(axis_dir, elevation=25):
    """Ekseni axis_dir yönünde, yarıçapı yükseklikle değişen bilinen bir parça ve siluetini üretir."""
    cam = Camera(scale=150, cx=400, cy=450, R=view_rotation(elevation))
    d = np.asarray(axis_dir, float)
    d /= np.linalg.norm(d)
    s = np.linspace(0, 2.0, 50)
    P = np.outer(s, d)
    r = 0.35 + 0.12 * np.sin(s * 2.0)
    V, F = sweep(P, r, n=64)
    return cam, P, r, render_mask(V, F, cam)


def test_walk_to_edge_on_rectangle():
    mask = np.zeros((100, 100), np.uint8)
    mask[40:60, 20:80] = 255
    assert walk_to_edge(mask, (50, 50), (1, 0)) == pytest.approx(30, abs=1)
    assert walk_to_edge(mask, (50, 50), (0, -1)) == pytest.approx(10, abs=1)
    assert walk_to_edge(mask, (5, 5), (1, 0)) is None


def test_smooth_keeps_constant():
    assert np.allclose(smooth([2, 2, 2, 2, 2]), 2)


@pytest.mark.parametrize("axis_dir", [[0, 0, 1], [0.4, 0, 1], [-0.3, 0.2, 1]])
def test_recovers_radii_and_length(axis_dir):
    cam, P_true, r_true, mask = make_scene(axis_dir)
    d = P_true[-1] / np.linalg.norm(P_true[-1])
    start = Circle3D(center=P_true[0], normal=d, radius=r_true[0])
    end_px = project(P_true[-1:], cam)[0]

    P, radii = snap_sweep(mask, start, end_px, cam, n_rings=30, smoothing=1)

    length = np.linalg.norm(P[-1] - P[0])
    assert length == pytest.approx(2.0, rel=0.05)
    s = np.linalg.norm(P - P[0], axis=1)
    expected = np.interp(s, np.linalg.norm(P_true, axis=1), r_true)
    assert np.allclose(radii[1:-1], expected[1:-1], atol=0.02)   # 0.02 birim = 3 piksel


def test_full_pipeline_from_strokes():
    # M4 + M5 birlikte: kullanıcının üç çizgisinden parça
    cam, P_true, r_true, mask = make_scene([0, 0, 1], elevation=30)
    view = cam.R[2]
    n = np.array([0, 0, 1.0])
    e = np.cross(n, view); e /= np.linalg.norm(e)
    f = np.cross(n, e)
    c0, r0 = P_true[0], r_true[0]
    a, b = project([c0 - r0 * e, c0 + r0 * e], cam)       # 1. çizgi
    m = project([c0 + r0 * f], cam)[0]                     # 2. çizgi
    end_px = project(P_true[-1:], cam)[0]                  # 3. çizginin sonu

    circle = circle_from_strokes(a, b, m, cam)
    P, radii = snap_sweep(mask, circle, end_px, cam, n_rings=30)
    V, F = sweep(P, radii)
    assert len(V) > 0 and radii.max() == pytest.approx(r_true.max(), abs=0.03)


def test_axis_toward_camera_raises():
    cam = Camera(scale=100, cx=50, cy=50)
    mask = np.full((100, 100), 255, np.uint8)
    circle = Circle3D(center=np.zeros(3), normal=cam.R[2].copy(), radius=0.2)
    with pytest.raises(ValueError):
        snap_sweep(mask, circle, (60, 50), cam)