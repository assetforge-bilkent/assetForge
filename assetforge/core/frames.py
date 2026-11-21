import numpy as np


def tangents(P):
    """Eksenin her noktasındaki yönü (boyu 1 olan vektör) döndürür."""
    P = np.asarray(P, dtype=float)
    T = np.gradient(P, axis=0)                          # her noktada ilerleme yönü
    lengths = np.linalg.norm(T, axis=1, keepdims=True)  # her okun uzunluğu
    return T / lengths                                  # boyu 1'e indir


def frames(P):
    """Eksenin her noktasında T (ileri), N (sağ), B (yukarı) yönlerini döndürür.

    N, bir önceki noktadan taşınarak hesaplanır; böylece halkalar burulmaz.
    """
    T = tangents(P)

    # İlk nokta: T'ye paralel olmayan bir yardımcı yön seç
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(T[0], helper)) > 0.9:       # T neredeyse z yönündeyse
        helper = np.array([1.0, 0.0, 0.0])    # x yönünü kullan
    n0 = np.cross(T[0], helper)
    N = [n0 / np.linalg.norm(n0)]

    # Sonraki noktalar: önceki N'yi yeni düzleme yatır
    for i in range(1, len(T)):
        n = N[-1] - np.dot(N[-1], T[i]) * T[i]   # T yönündeki parçayı çıkar
        N.append(n / np.linalg.norm(n))

    N = np.array(N)
    B = np.cross(T, N)
    return T, N, B
if __name__ == "__main__":
    P = [[0, 0, 0], [0, 0, 1], [0, 1, 1]]
    T, N, B = frames(P)
    print("T:\n", T)
    print("N:\n", N)
    print("B:\n", B)
    # Kontrol: üç yön birbirine dik olmalı, yani nokta çarpımları 0 olmalı
    print("T·N:", np.sum(T * N, axis=1).round(6))
    print("T·B:", np.sum(T * B, axis=1).round(6))