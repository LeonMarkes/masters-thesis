import numpy as np
from PIL import Image
from kerneli import konvolucijski_filteri, detekcija_ruba
from typing import List
from util import relu, softmax
import sys


class Konvolucijska_neuronska_mreza:

    def __init__(self,
                 podaci: List[np.ndarray],
                 skriveni_sloj: int,
                 izlazni_sloj: int) -> None:
        self.podaci: List[np.ndarray] = podaci
        self.skriveni_sloj = np.zeros(skriveni_sloj)
        self.izlazni_sloj = np.zeros(izlazni_sloj)
        self.tf_ss: np.ndarray = None
        self.tf_is: np.ndarray = None
        self.o_ss = None
        self.o_is = None
        self.relu_f = lambda x: x * (x > 0)

    def ucitaj_sliku(self, naziv_slike: str) -> Image:  # vraća crno bijelu sliku
        return Image.open(naziv_slike, 'r').convert('L')

    def pretvori_u_niz(self, vrijednost) -> np.ndarray:
        return np.asarray(vrijednost, dtype=float)

    def pretvori_u_sliku(self, vrijednost: List[List[float]]) -> Image:
        niz_znakova: np.ndarray = self.pretvori_u_niz(vrijednost)
        return Image.fromarray(niz_znakova)

    def provjeri_najvecu_vrijednost(self, pikseli: np.ndarray) -> float:
        return np.max(pikseli)

    def provjeri_srednju_vrijednost(self, pikseli: np.ndarray) -> float:
        return np.mean(pikseli)

    def udruzivanje_slike(self, mapa_znacajki: np.ndarray,
                          velicina: int = 2,
                          pomak: int = 2) -> np.ndarray:  # Pooling 2x2
        udruzena_slika = np.zeros((np.uint16((mapa_znacajki.shape[0] - velicina + 1) / pomak),
                                   np.uint16((mapa_znacajki.shape[1] - velicina + 1) / pomak),
                                   mapa_znacajki.shape[-1]))
        for broj_mape in range(mapa_znacajki.shape[-1]):
            red_udruzene_slike = 0
            for red in np.arange(0, mapa_znacajki.shape[0] - velicina - 1, pomak):
                stupac_udruzene_slike = 0
                for stupac in np.arange(0, mapa_znacajki.shape[1] - velicina - 1, pomak):
                    udruzena_slika[
                        red_udruzene_slike, stupac_udruzene_slike, broj_mape] = self.provjeri_najvecu_vrijednost(
                        [mapa_znacajki[red:red + velicina, stupac:stupac + velicina, broj_mape]])
                    stupac_udruzene_slike += 1
                red_udruzene_slike += 1
        return udruzena_slika

    def primjeni_kernel(self, tri_x_tri: np.ndarray, kernel: np.ndarray = detekcija_ruba) -> float:
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

    def konvolucija(self, slika: np.ndarray, konvolucijski_filteri: np.ndarray) -> np.ndarray:
        # if len(slika.shape) > 2 or len(konvolucijski_filteri.shape) > 3:  # Provjere da li su dane vrijednosti ispravne
        #     if slika.shape[-1] != konvolucijski_filteri.shape[-1]:
        #         print('Error: Slika i filter nisu istih dimenzija.', slika.shape, konvolucijski_filteri.shape)
        #         sys.exit()
        # if konvolucijski_filteri.shape[1] != konvolucijski_filteri.shape[2]:
        #     print('Error: Matrice nisu kvadratne')
        #     sys.exit()
        # if konvolucijski_filteri.shape[1] % 2 == 0:
        #     print('Error: Matrica nema neparni broj vrijednosti.')
        #     sys.exit()

        mapa_znacajki = np.zeros((slika.shape[0] - konvolucijski_filteri.shape[1] + 1,
                                  slika.shape[1] - konvolucijski_filteri.shape[1] + 1,
                                  konvolucijski_filteri.shape[0]))
        for broj_filtera in range(konvolucijski_filteri.shape[0]):
            # print('Filter ', broj_filtera + 1)
            trenutni_filter = konvolucijski_filteri[broj_filtera, :]  # dohvačanje filtera
            if len(trenutni_filter.shape) > 2:
                konvolucijska_mapa = self.trazenje_znacajki(slika, trenutni_filter)
                for broj_kanala in range(1, trenutni_filter.shape[-1]):
                    konvolucijska_mapa = konvolucijska_mapa + self.primjeni_kernel(
                        slika[:, :, broj_kanala], trenutni_filter[:, :, broj_kanala])
            else:
                konvolucijska_mapa = self.trazenje_znacajki(slika, trenutni_filter)
            mapa_znacajki[:, :, broj_filtera] = konvolucijska_mapa
        return self.relu_f(mapa_znacajki)

    def relu(self, mapa_znacajki: np.ndarray) -> np.ndarray:
        izlazni_relu = np.zeros(mapa_znacajki.shape)
        if len(mapa_znacajki) == 1:
            for stavka in range(mapa_znacajki.shape[0]):
                print(mapa_znacajki[stavka])
                izlazni_relu[stavka] = relu(mapa_znacajki[stavka])
        else:
            for broj_mape in range(mapa_znacajki.shape[-1]):
                for red in np.arange(0, mapa_znacajki.shape[0]):
                    for stupac in np.arange(0, mapa_znacajki.shape[1]):
                        izlazni_relu[red, stupac, broj_mape] = relu(mapa_znacajki[red, stupac, broj_mape])
            return izlazni_relu

    def pretvori_u_1d_niz(self) -> List[float]:  # vraća jednodimenzionalni niz za ulaz u neuronsku mrežu
        return self.znacajke.flatten().tolist()

    def konvolucijski_sloj(self):
        if self.udruzena_relu_slika is None:
            mapa_znacajki: np.ndarray = self.konvolucija(self.slika, konvolucijski_filteri)
            mapa_relu_znacajki: np.ndarray = self.relu(mapa_znacajki)
            self.udruzena_relu_slika = self.udruzivanje_slike(mapa_relu_znacajki)
        else:
            znacajke: np.ndarray = np.zeros((
                np.uint16((self.udruzena_relu_slika.shape[0] - 3) / 2),
                np.uint16((self.udruzena_relu_slika.shape[1] - 3) / 2),
                self.udruzena_relu_slika.shape[-1] * 2
            ))
            brojac: int = 0
            for slika in range(self.udruzena_relu_slika.shape[-1]):
                mapa_znacajki = self.konvolucija(self.udruzena_relu_slika[:, :, slika], konvolucijski_filteri)
                mapa_relu_znacajki = self.relu(mapa_znacajki)
                udruzena_relu_slike = self.udruzivanje_slike(mapa_relu_znacajki)
                for broj_slike in range(udruzena_relu_slike.shape[-1]):
                    znacajke[:, :, brojac] = udruzena_relu_slike[:, :, broj_slike]
                    brojac += 1
            self.udruzena_relu_slika = znacajke

    def pozivanje_konvolucijskog_sloja(self, iteracija: int) -> None:
        for _ in range(iteracija):
            self.konvolucijski_sloj()

    def zgusnjavanje_izlaza(self, velicina: int = 1024) -> List[float]:
        relu_niz_1d: List[float] = self.udruzena_relu_slika.ravel().tolist()
        iteracija: np.uint8 = np.uint8(len(relu_niz_1d) / velicina)
        zgusnuti_niz: List[float] = []
        for i in range(velicina):
            zgusnuti_niz.append(sum(relu_niz_1d[:iteracija]))
            relu_niz_1d = relu_niz_1d[iteracija:]
        return zgusnuti_niz

    def treniranje(self, velicina_skupa: int) -> None:
        skup_za_ucenje: np.ndarray = self.podaci[:velicina_skupa]
        skup_za_testiranje: np.ndarray = self.podaci[velicina_skupa:]
        m:int = len(skup_za_ucenje)
        tezinski_faktori = False
        for parametri, oznaka in skup_za_ucenje:
            mape_znacajki: np.ndarray = self.konvolucija(parametri, detekcija_ruba)
            umanjene_mape: np.ndarray = self.udruzivanje_slike(mape_znacajki)

            izravnati_niz = umanjene_mape.reshape(-1, 1)

            if tezinski_faktori:
                # weights
                pass
            else:
                n_x = izravnati_niz.shape[0]
                n_h = self.skriveni_sloj.shape[0]
                n_y = self.izlazni_sloj.shape[0]
                print(n_x, n_h, n_y)

                self.tf_ss = np.random.randn(n_h, n_x) * .01
                self.o_ss = np.zeros((n_h, 1))
                self.tf_is = np.random.randn(n_y, n_h) * .01
                self.o_is = np.zeros((n_y, 1))
                tezinski_faktori = True

            Z1 = np.dot(self.tf_ss, izravnati_niz)
            print(Z1.shape)
            A1 = self.relu_f(Z1)
            print(A1.shape)
            Z2 = np.dot(self.tf_is, A1)
            print(Z2.shape)
            A2 = softmax(Z2)
            print(A2.shape)

            logprobs = np.multiply(np.log(A2), oznaka) + np.multiply((1 - oznaka), np.log(1 - A2))
            cost = - np.sum(logprobs) / m

            print(A2, oznaka)
            dZ2 = A2.T - oznaka
            dW2 = (1 / m) * np.dot(dZ2, A1.T)
            db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
            dZ1 = np.multiply(np.dot(self.tf_is.T, dZ2), 1 - np.power(A1, 2))
            dW1 = (1 / m) * np.dot(dZ1, izravnati_niz.T)
            db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)







            break



