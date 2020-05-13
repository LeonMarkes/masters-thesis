import numpy as np
from PIL import Image
from kerneli import konvolucijski_kernel, konvolucijski_filteri
from typing import List
import sys


class Konvolucijska_neuronska_mreza():

    def __init__(self, naziv_slike: str) -> None:
        self.slika: np.ndarray = self.pretvori_u_niz(self.ucitaj_sliku(naziv_slike))
        self.mapa_znacajki: np.ndarray = self.konvolucija(self.slika, konvolucijski_filteri)
        self.mapa_znacajki_relu = self.relu(self.mapa_znacajki)
        # self.udruzena_slika: Image = self.udruzi_sliku()
        # self.znacajke: np.ndarray = self.trazenje_znacajki()

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

    def primjeni_kernel(self, tri_x_tri: np.ndarray, kernel: np.ndarray = konvolucijski_kernel) -> float:
        return np.sum(np.multiply(tri_x_tri, kernel))

    def trazenje_znacajki(self, slika: np.ndarray, filter: np.ndarray) -> np.ndarray:
        znacajke: List[List[float]] = []
        for i in range(1, slika.shape[0] - 1):
            red_znacajki: List[float] = []
            for j in range(1, slika.shape[1] - 1):
                tri_x_tri: np.ndarray[float] = np.array([
                    [slika[i - 1][j - 1], slika[i - 1][j], slika[i - 1][j + 1]],
                    [slika[i][j - 1], slika[i][j], slika[i][j + 1]],
                    [slika[i + 1][j - 1], slika[i + 1][j], slika[i + 1][j + 1]]
                ])
                znacajka: float = self.primjeni_kernel(tri_x_tri, filter)
                red_znacajki.append(znacajka)
            znacajke.append(red_znacajki)

        return self.pretvori_u_niz(znacajke)

    def konvolucija(self, slika: Image, konvolucijski_filteri: np.ndarray) -> np.ndarray:
        if len(slika.shape) > 2 or len(konvolucijski_filteri.shape) > 3: # Provjere da li su dane vrijednosti ispravne
            if slika.shape[-1] != konvolucijski_filteri.shape[-1]:
                print('Error: Slika i filter nisu istih dimenzija.')
                sys.exit()
        if konvolucijski_filteri.shape[1] != konvolucijski_filteri.shape[2]:
            print('Error: Matrice nisu kvadratne')
            sys.exit()
        if konvolucijski_filteri.shape[1] % 2 == 0:
            print('Error: Matrica nema neparni broj vrijednosti.')
            sys.exit()

        mapa_znacajki = np.zeros((slika.shape[0] - konvolucijski_filteri.shape[1] + 1,
                                  slika.shape[1] - konvolucijski_filteri.shape[1] + 1,
                                  konvolucijski_filteri.shape[0]))
        for broj_filtera in range(konvolucijski_filteri.shape[0]):
            print('Filter ', broj_filtera + 1)
            trenutni_filter = konvolucijski_filteri[broj_filtera, :] # dohvačanje filtera
            print(trenutni_filter)
            if len(trenutni_filter.shape) > 2:
                konvolucijska_mapa = self.primjeni_kernel(slika[:, :, 0], trenutni_filter[:, :, 0])
                print(konvolucijska_mapa)
                for broj_kanala in range(1, trenutni_filter.shape[-1]):
                    konvolucijska_mapa = konvolucijska_mapa + self.primjeni_kernel(
                        slika[:, :, broj_kanala], trenutni_filter[:, :, broj_kanala])
            else:
                konvolucijska_mapa = self.trazenje_znacajki(slika, trenutni_filter)
            mapa_znacajki[:, :, broj_filtera] = konvolucijska_mapa
        return mapa_znacajki

    def relu(self, mapa_znacajki: np.float_) -> np.ndarray:
        izlazni_relu = np.zeros(mapa_znacajki.shape)
        for broj_mape in range(mapa_znacajki.shape[-1]):
            for red in np.arange(0, mapa_znacajki.shape[0]):
                for stupac in np.arange(0, mapa_znacajki.shape[1]):
                    izlazni_relu[red, stupac, broj_mape] = np.max([mapa_znacajki[red, stupac, broj_mape], 0])
        return izlazni_relu

    def pretvori_u_1d_niz(self) -> List[float]:  # vraća jednodimenzionalni niz za ulaz u neuronsku mrežu
        return self.znacajke.ravel().tolist()


konvo: Konvolucijska_neuronska_mreza = Konvolucijska_neuronska_mreza('test3.png')
x = konvo.mapa_znacajki_relu
print(x)
# normalizacija_skaliranih_znacajki(x)
# print(x.ravel().tolist())
