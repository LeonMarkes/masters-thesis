import numpy as np
from PIL import Image
from kerneli import konvolucijski_kernel, horizontalni_kernel, vertikalni_kernel
from udruzivanje_slika import udruzi_sliku
from typing import List


def konvolucija(tri_x_tri: List[List[float]], kernel: np.ndarray = konvolucijski_kernel) -> float:
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


with Image.open('bambi.png') as slika:
    Image.fromarray(trazenje_znacajki(udruzi_sliku(slika))).show()
    slika.show()
    # print(trazenje_svojstava(slika))




    def trazenje_znacajki(self) -> np.ndarray:
        slika: np.ndarray = self.pretvori_u_niz(self.udruzena_slika)
        konvoluirane_znacajke: List[List[float]] = []
        for i in range(1, slika.shape[0] - 1):
            red_znacajki: List[float] = []
            for j in range(1, slika.shape[1] - 1):
                # tri_x_tri: np.ndarray[float] = np.array([
                #     [slika[i - 1][j - 1], slika[i - 1][j], slika[i - 1][j + 1]],
                #     [slika[i][j - 1], slika[i][j], slika[i][j + 1]],
                #     [slika[i + 1][j - 1], slika[i + 1][j], slika[i + 1][j + 1]]
                # ])
                znacajka: float = self.primjeni_kernel(tri_x_tri)
                red_znacajki.append(znacajka)
            konvoluirane_znacajke.append(red_znacajki)

        return self.pretvori_u_niz(konvoluirane_znacajke)

    def udruzi_sliku(self, mape_znacajki: np.ndarray) -> Image:  # Pooling 2x2
        udruzene_slike: List[List[List[float]]] = []
        for indeks, znacajka in enumerate(mape_znacajki[-1]):
            udruzena_slika: List[List[float]] = []
            for i in range(0, znacajka.shape[0] - 1, 2):
                udruzeni_pikseli: List[float] = []
                for j in range(0, znacajka.shape[1] - 1, 2):
                    pikseli: np.ndarray[float] = np.array((znacajka[i][j], znacajka[i][j + 1],
                                                           znacajka[i + 1][j], znacajka[i + 1][j + 1]))
                    udruzen_piksel: float = self.provjeri_najvecu_vrijednost(pikseli)
                    udruzeni_pikseli.append(udruzen_piksel)
                udruzena_slika.append(udruzeni_pikseli)
            udruzene_slike.append(udruzena_slika)
        nova_slika = self.pretvori_u_niz(udruzene_slike)
        return nova_slika