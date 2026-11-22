import numpy as np
import pytest

from assetforge.core.camera import Camera, project, view_rotation
from assetforge.core.cross_section import circle_from_strokes, circle_points


def strokes_for(center, normal, radius, cam):
    """Bilinen bir 3D daire için, kullanıcının çizeceği iki çizgiyi hesaplar."""
    view = cam.R[2]
    e = np.cross(normal, view)
    e /= np.linalg.norm(e)                     # dairenin içinde, bakışa dik: tam boyda görünür
    f = np.cross(normal, e)                    # dairenin içinde, en çok kısalan yön
    a, b = project([center - radius * e, center + radius * e], cam)
    m = project([center + radius * f], cam)[0]
    return a, b, m


def angle_between(u, v):
    c = abs(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def test_face_on_circle():
    cam = Camera(scale=100, cx=400, cy=300)                 # tam önden
    a, b, m = [300, 300], [500, 300], [400, 200]            # uzun = kısa: daire
    c = circle_from_strokes(a, b, m, cam)
    assert c.radius == pytest.approx(1.0)
    assert angle_between(c.normal, cam.R[2]) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("elevation", [15, 30, 45, 70])
def test_recovers_horizontal_circle_seen_from_above(elevation):
    cam = Camera(scale=120, cx=400, cy=300, R=view_rotation(elevation))
    center, normal, radius = np.array([0.3, 0, 1.0]), np.array([0, 0, 1.0]), 0.8
    a, b, m = strokes_for(center, normal, radius, cam)
    c = circle_from_strokes(a, b, m, cam)
    assert c.radius == pytest.approx(radius)
    assert angle_between(c.normal, normal) < 0.5            # derece


def test_flip_gives_the_other_solution():
    cam = Camera(scale=120, cx=400, cy=300, R=view_rotation(30))
    a, b, m = strokes_for(np.zeros(3), np.array([0, 0, 1.0]), 1.0, cam)
    c1 = circle_from_strokes(a, b, m, cam)
    c2 = circle_from_strokes(a, b, m, cam, flip=True)
    # ikisi de aynı elipsi verir ama normalleri farklıdır
    assert angle_between(c1.normal, c2.normal) > 10
    for c in (c1, c2):
        uv = project(circle_points(c, 360), cam)
        assert np.ptp(uv[:, 0]) == pytest.approx(240, rel=1e-2)          # uzun eksen 2 * 120
        assert np.ptp(uv[:, 1]) == pytest.approx(240 * 0.5, rel=1e-2)    # kısa = uzun * sin(30)


def test_too_short_stroke_raises():
    cam = Camera(scale=100, cx=0, cy=0)
    with pytest.raises(ValueError):
        circle_from_strokes([10, 10], [10.2, 10], [10, 5], cam)