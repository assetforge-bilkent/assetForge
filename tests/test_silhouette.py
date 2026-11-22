import cv2
import numpy as np
import pytest

from assetforge.image.silhouette import (axis_and_radii_from_mask, clean_mask, grabcut_mask,
                                         object_mask, row_profile)


def vase_silhouette(size=(400, 300), px_per_unit=300):
    """Yarıçapı bilinen bir vazonun önden görünen siluetini çizer (doğru cevabı biliyoruz)."""
    h, w = size
    z = np.linspace(0, 1, 60)
    r = 0.15 + 0.08 * np.sin(z * 1.5 * np.pi)               # dünya biriminde yarıçap
    cx, bottom = w / 2, h - 50
    left = np.c_[cx - r * px_per_unit, bottom - z * px_per_unit]
    right = np.c_[cx + r * px_per_unit, bottom - z * px_per_unit]
    outline = np.vstack([left, right[::-1]]).round().astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [outline], 255)
    return mask, z, r


def test_row_profile_of_rectangle():
    mask = np.zeros((10, 20), np.uint8)
    mask[2:8, 5:15] = 255                       # 10 piksel genişliğinde dikdörtgen
    rows, centers, radii = row_profile(mask)
    assert rows.tolist() == list(range(2, 8))
    assert np.allclose(centers, 9.5)
    assert np.allclose(radii, 5)


def test_recovers_known_vase_radii():
    mask, z_true, r_true = vase_silhouette()
    P, radii = axis_and_radii_from_mask(mask, n_rings=30, height=1.0, smooth=1)
    expected = np.interp(P[:, 2], z_true, r_true)
    # uçlardaki birkaç halka hariç, piksel hatası kadar (1/300) yakın olmalı
    assert np.allclose(radii[2:-2], expected[2:-2], atol=0.01)
    assert P[0, 2] == pytest.approx(0) and P[-1, 2] == pytest.approx(1)


def test_clean_mask_keeps_biggest_piece():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 30:70] = 255                    # asıl nesne
    mask[5:8, 5:8] = 255                        # kırıntı
    cleaned = clean_mask(mask)
    assert cleaned[5:8, 5:8].max() == 0
    assert cleaned[50, 50] == 255


def overlap(mask, mask_true):
    inter = np.logical_and(mask > 0, mask_true > 0).sum()
    union = np.logical_or(mask > 0, mask_true > 0).sum()
    return inter / union


def test_object_mask_on_gray_background():
    mask_true, _, _ = vase_silhouette()
    img = np.full((*mask_true.shape, 3), 235, np.uint8)     # açık gri duvar
    img[mask_true > 0] = (40, 90, 160)                      # kahverengimsi vazo
    assert overlap(object_mask(img), mask_true) > 0.95


def test_object_mask_on_colored_background_with_white_object():
    mask_true, _, _ = vase_silhouette()
    img = np.zeros((*mask_true.shape, 3), np.uint8)
    img[:] = (70, 80, 200)                                  # kırmızı zemin (BGR)
    img[mask_true > 0] = (225, 225, 225)                    # beyaz vazo
    assert overlap(object_mask(img), mask_true) > 0.95


def test_grabcut_on_plain_background():
    mask_true, _, _ = vase_silhouette()
    img = np.full((*mask_true.shape, 3), 235, np.uint8)
    img[mask_true > 0] = (40, 90, 160)
    assert overlap(grabcut_mask(img), mask_true) > 0.9


def test_clean_mask_fills_holes():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 255
    mask[45:55, 45:55] = 0                      # ortada delik (desen gibi)
    assert clean_mask(mask)[50, 50] == 255


def test_empty_mask_raises():
    with pytest.raises(ValueError):
        axis_and_radii_from_mask(np.zeros((50, 50), np.uint8))
