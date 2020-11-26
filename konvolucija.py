import numpy as np
from PIL import Image
from kerneli import vertikalni_kernel, horizontalni_kernel, detekcija_ruba
from udruzivanje_slika import udruzi_sliku
from typing import List


def konvolucija(tri_x_tri: List[List[float]], kernel: np.ndarray = detekcija_ruba) -> float:
    umnozak_dviju_matrica = np.multiply(tri_x_tri, kernel)
    return umnozak_dviju_matrica.sum()


def trazenje_znacajki(slika: Image) -> np.ndarray:
    slika: np.ndarray = np.asarray(slika.convert('L'), dtype=float)
    konvolirane_znacajke: List[List[float]] = []
    for i in range(1, slika.shape[0] - 1):
        red_znacajki: List[float] = []
        for j in range(1, slika.shape[1] - 1):
            tri_x_tri: np.ndarray[float] = [
                [slika[i - 1][j - 1], slika[i - 1][j], slika[i - 1][j + 1]],
                [slika[i][j - 1], slika[i][j], slika[i][j + 1]],
                [slika[i + 1][j - 1], slika[i + 1][j], slika[i + 1][j + 1]]
            ]
            znacajka: float = konvolucija(tri_x_tri)
            red_znacajki.append(znacajka)
        konvolirane_znacajke.append(red_znacajki)

    konvolirane_znacajke = np.asarray(konvolirane_znacajke)
    return konvolirane_znacajke


with Image.open('frey/dog.1.png') as slika:
    Image.fromarray(trazenje_znacajki(slika)).show()
    slika.show()




