import numpy as np
from PIL import Image
from kerneli import konvolucijski_kernel, konvolucijski_filteri
from typing import List
import sys


class Konvolucijska_neuronska_mreza():

    def __init__(self, naziv_slike: str, kernel: np.ndarray = konvolucijski_kernel) -> None:
        self.slika: Image = self.ucitaj_sliku(naziv_slike)
        self.mapa_svojstava: np.ndarray = self.konv(self.slika, konvolucijski_filteri)
        self.mapa_svojstava_relu = self.relu(self.mapa_svojstava)
        self.kernel: np.ndarray = kernel
        self.udruzena_slika: Image = self.udruzi_sliku()
        self.znacajke: np.ndarray = self.trazenje_znacajki()

    def ucitaj_sliku(self, naziv_slike: str) -> Image:  # vraća crno bijelu sliku
        return Image.open(naziv_slike, 'r').convert('L')

    def pretvori_u_niz(self, vrijednost) -> np.ndarray:
        return np.asarray(vrijednost, dtype=float)

    def pretvori_u_sliku(self, vrijednost: List[List[float]]) -> Image:
        niz_znakova: np.ndarray = self.pretvori_u_niz(vrijednost)
        return Image.fromarray(niz_znakova)

    def provjeri_najvecu_vrijednost(self, pikseli: np.ndarray) -> float:
        return pikseli.max()

    def provjeri_srednju_vrijednost(self, pikseli: np.ndarray) -> float:
        return pikseli.mean()

    def udruzi_sliku(self) -> Image:  # Pooling 2x2
        slika: np.ndarray = self.pretvori_u_niz(self.slika)
        udruzena_slika: List[List[float]] = []
        for i in range(0, slika.shape[0] - 1, 2):
            udruzeni_pikseli: List[float] = []
            for j in range(0, slika.shape[1] - 1, 2):
                pikseli: np.ndarray[float] = np.array((slika[i][j], slika[i][j + 1],
                                                       slika[i + 1][j], slika[i + 1][j + 1]))
                udruzen_piksel: float = self.provjeri_najvecu_vrijednost(pikseli)
                udruzeni_pikseli.append(udruzen_piksel)
            udruzena_slika.append(udruzeni_pikseli)

        nova_slika = self.pretvori_u_sliku(udruzena_slika)
        return nova_slika

    def konvolucija(self, tri_x_tri: np.ndarray, kernel: np.ndarray = konvolucijski_kernel) -> float:
        return np.multiply(tri_x_tri, kernel).sum()

    def trazenje_znacajki(self) -> np.ndarray:
        slika: np.ndarray = self.pretvori_u_niz(self.udruzena_slika)
        konvoluirane_znacajke: List[List[float]] = []
        for i in range(1, slika.shape[0] - 1):
            red_znacajki: List[float] = []
            for j in range(1, slika.shape[1] - 1):
                tri_x_tri: np.ndarray[float] = np.array([
                    [slika[i - 1][j - 1], slika[i - 1][j], slika[i - 1][j + 1]],
                    [slika[i][j - 1], slika[i][j], slika[i][j + 1]],
                    [slika[i + 1][j - 1], slika[i + 1][j], slika[i + 1][j + 1]]
                ])
                znacajka: float = self.konvolucija(tri_x_tri)
                red_znacajki.append(znacajka)
            konvoluirane_znacajke.append(red_znacajki)

        return self.pretvori_u_niz(konvoluirane_znacajke)

    def konv(self, slika: Image, konvolucijski_filteri: np.ndarray) -> np.ndarray:
        if len(slika.shape) > 2 or len(konvolucijski_filteri) > 3:
            print('Error: Nisu unesene dobre dimenzije filtera i slike.')
            sys.exit()
        if slika.shape[-1] != konvolucijski_filteri.shape[-1]:
            print('Error: Slika i filter nisu iste dimenzije.')
            sys.exit()
        if konvolucijski_filteri.shape[1] != konvolucijski_filteri.shape[2]:
            print('Error: Filter nije kvadratna matrica.')
            sys.exit()
        if konvolucijski_filteri.shape[1] % 2 == 0:
            print('Error: Filteri moraju imati neparni broj vrijednosti.')
            sys.exit()

        mapa_znacajki = np.zeros((slika.shape[0] - konvolucijski_filteri.shape[1] + 1,
                                  slika.shape[1] - konvolucijski_filteri.shape[1] + 1,
                                  konvolucijski_filteri.shape[0]))

        for broj_filtera in range(konvolucijski_filteri.shape[0]):
            print('Filter ', broj_filtera + 1)
            trenutni_filter = konvolucijski_filteri[broj_filtera, :]
            if len(trenutni_filter.shape) > 2:
                konvolucijska_mapa = self.konvolucija(slika[:, :, 0], trenutni_filter[:, :, 0])
                for ch_num in range(1, trenutni_filter.shape[-1]):
                    konvolucijska_mapa = konvolucijska_mapa + self.konvolucija(
                        slika[:, :, ch_num], trenutni_filter[:, :, ch_num])
            else:
                konvolucijska_mapa = self.konvolucija(slika, trenutni_filter)
            mapa_znacajki[:, :, broj_filtera] = konvolucijska_mapa
        return mapa_znacajki



    def pretvori_u_1d_niz(self) -> List[float]:  # vraća jednodimenzionalni niz za ulaz u neuronsku mrežu
        return self.znacajke.ravel().tolist()


# konv: Konvolucijska_neuronska_mreza = Konvolucijska_neuronska_mreza('bambi.png')
# print(konv.pretvori_u_1d_niz())
# print(len(konv.pretvori_u_1d_niz()))
