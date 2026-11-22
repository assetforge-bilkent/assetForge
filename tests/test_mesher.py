from collections import Counter

import numpy as np
import pytest

from assetforge.core.mesher import cylinder, grid_faces, save_obj


def edge_counts(F):
    """Her kenarın kaç üçgen tarafından paylaşıldığını sayar."""
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return Counter(map(tuple, np.sort(edges, axis=1)))


def signed_volume(V, F):
    """Mesh'in hacmi. Yüzeyler dışa bakıyorsa pozitif çıkar."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0


def test_grid_faces_small_example():
    # Kağıt üzerinde çözdüğümüz 2 halka x 4 nokta örneği
    expected = [[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    assert grid_faces(2, 4).tolist() == expected


def test_counts():
    V, F = cylinder(n=36, rings=12)
    assert V.shape == (36 * 12 + 2, 3)
    assert F.shape == (2 * 36 * 11 + 2 * 36, 3)


def test_no_holes():
    # Kapalı bir yüzeyde her kenar tam 2 üçgen tarafından paylaşılır
    V, F = cylinder(n=24, rings=6)
    assert all(count == 2 for count in edge_counts(F).values())


def test_faces_point_outward_and_volume_is_right():
    r, h = 1.5, 3.0
    V, F = cylinder(radius=r, height=h, n=256, rings=4)
    vol = signed_volume(V, F)
    assert vol > 0, "Hacim negatif: üçgenler içe bakıyor"
    assert vol == pytest.approx(np.pi * r * r * h, rel=1e-3)


def test_save_obj(tmp_path):
    V, F = cylinder(n=8, rings=2)
    path = tmp_path / "c.obj"
    save_obj(str(path), V, F)
    lines = path.read_text().splitlines()
    assert sum(l.startswith("v ") for l in lines) == len(V)
    assert sum(l.startswith("f ") for l in lines) == len(F)