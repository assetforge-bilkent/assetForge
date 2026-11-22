import numpy as np
import pytest

from assetforge.core.camera import Camera, depth, mesh_edges, project, unproject, view_rotation
from assetforge.core.mesher import cylinder


def test_rotation_is_orthonormal():
    for e in (0, 30, 60, 90):
        R = view_rotation(e)
        assert np.allclose(R @ R.T, np.eye(3))
        assert np.isclose(np.linalg.det(R), 1.0)


def test_front_view_known_points():
    cam = Camera(scale=100, cx=320, cy=240)
    X = [[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 5, 0]]
    uv = project(X, cam)
    assert np.allclose(uv[0], [320, 240])     # orijin -> (cx, cy)
    assert np.allclose(uv[1], [420, 240])     # x yönü -> sağa
    assert np.allclose(uv[2], [320, 140])     # z yukarısı -> resimde yukarı (v azalır)
    assert np.allclose(uv[3], [320, 240])     # tam önden bakınca derinlik görünmez


def test_circle_seen_from_above_is_an_ellipse():
    # Yatay bir çember, 30 derece yukarıdan bakınca elips görünür:
    # kısa eksen / uzun eksen = sin(30 derece) = 0.5
    a = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    circle = np.c_[np.cos(a), np.sin(a), np.zeros_like(a)]
    uv = project(circle, Camera(scale=100, cx=0, cy=0, R=view_rotation(30)))
    width = np.ptp(uv[:, 0])
    height = np.ptp(uv[:, 1])
    assert height / width == pytest.approx(0.5, abs=1e-3)


def test_unproject_reverses_project():
    cam = Camera(scale=80, cx=400, cy=300, R=view_rotation(25))
    X = np.random.default_rng(0).normal(size=(50, 3))
    back = unproject(project(X, cam), depth(X, cam), cam)
    assert np.allclose(back, X)


def test_mesh_edges_of_cylinder():
    V, F = cylinder(n=8, rings=3)
    E = mesh_edges(F)
    # Euler formülü kapalı yüzey için: V - E + F = 2
    assert len(V) - len(E) + len(F) == 2
