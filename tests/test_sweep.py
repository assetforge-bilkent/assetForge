from collections import Counter

import numpy as np
import pytest

from assetforge.core.frames import frames, tangents
from assetforge.core.mesher import cylinder
from assetforge.core.sweep import arc_axis, sweep


def edge_counts(F):
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return Counter(map(tuple, np.sort(edges, axis=1)))


def signed_volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0


def straight_axis(k=10, height=2.0):
    return np.c_[np.zeros(k), np.zeros(k), np.linspace(0, height, k)]


# ---------- frames ----------

def test_tangents_of_straight_axis_point_up():
    T = tangents(straight_axis())
    assert np.allclose(T, [0, 0, 1])


def test_frames_are_orthonormal_on_a_curve():
    T, N, B = frames(arc_axis(steps=40))
    for v in (T, N, B):
        assert np.allclose(np.linalg.norm(v, axis=1), 1)   # boyları 1
    assert np.allclose(np.sum(T * N, axis=1), 0)           # birbirine dik
    assert np.allclose(np.sum(T * B, axis=1), 0)
    assert np.allclose(np.sum(N * B, axis=1), 0)


def test_no_twist_on_planar_curve():
    # Eksen tek bir düzlemde (xz) kıvrılıyorsa, halka o düzleme dik yön (y)
    # etrafında hiç dönmemeli: N ya da B'den biri baştan sona aynı kalmalı
    T, N, B = frames(arc_axis(steps=40))
    assert np.allclose(N, N[0]) or np.allclose(B, B[0])


# ---------- sweep ----------

def test_straight_sweep_matches_cylinder():
    V, F = sweep(straight_axis(k=12), radii=1.0, n=36)
    Vc, Fc = cylinder(radius=1.0, height=2.0, n=36, rings=12)
    assert V.shape == Vc.shape and F.shape == Fc.shape
    assert signed_volume(V, F) == pytest.approx(signed_volume(Vc, Fc), rel=1e-6)


def test_bent_pipe_is_closed_and_has_right_volume():
    r, R = 0.3, 2.0                                   # boru yarıçapı, dirsek yarıçapı
    V, F = sweep(arc_axis(radius=R, angle_deg=90, steps=60), radii=r, n=128)
    assert all(count == 2 for count in edge_counts(F).values())
    length = R * np.pi / 2                            # çeyrek çemberin uzunluğu
    assert signed_volume(V, F) == pytest.approx(np.pi * r * r * length, rel=1e-2)


def test_radius_per_point():
    k = 20
    radii = np.linspace(0.2, 1.0, k)
    V, F = sweep(straight_axis(k), radii=radii, n=16, caps=False)
    rings = V.reshape(k, 16, 3)
    measured = np.linalg.norm(rings[:, :, :2], axis=2).mean(axis=1)
    assert np.allclose(measured, radii)