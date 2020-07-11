import numpy as np
from kerneli import konvolucijski_filteri, detekcija_ruba, sobel_filteri
from typing import List, Tuple
from util import relu, softmax
import matplotlib.pyplot as plt
from tqdm import tqdm


class Konvolucijska_neuronska_mreza:

    def __init__(self,
                 podaci: List[np.ndarray],
                 skriveni_sloj: int,
                 izlazni_sloj: int,
                 stopa_ucenja: float) -> None:
        self.podaci: List[np.ndarray] = podaci
        self.skriveni_sloj = skriveni_sloj
        self.izlazni_sloj = izlazni_sloj
        self.tf_ss: np.ndarray = None
        self.tf_is: np.ndarray = None
        self.o_ss = None
        self.o_is = None
        self.stopa_ucenja = stopa_ucenja

    def pretvori_u_niz(self, vrijednost) -> np.ndarray:
        return np.asarray(vrijednost, dtype=float)

    def provjeri_najvecu_vrijednost(self, pikseli: np.ndarray) -> float:
        return np.max(pikseli)

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

    def primjeni_kernel(self, tri_x_tri: np.ndarray, kernel: np.ndarray) -> float:
        return np.sum(np.multiply(tri_x_tri, kernel))

    def trazenje_znacajki(self, slika: np.ndarray, filter: np.ndarray) -> np.ndarray:
        znacajke: List[List[List[float]]] = []
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
        mapa_znacajki = np.zeros((slika.shape[0] - konvolucijski_filteri.shape[1] + 1,
                                  slika.shape[1] - konvolucijski_filteri.shape[1] + 1,
                                  konvolucijski_filteri.shape[0]))
        for broj_filtera in range(konvolucijski_filteri.shape[0]):
            trenutni_filter = konvolucijski_filteri[broj_filtera, :]  # dohvačanje filtera
            if len(slika.shape) > 2:
                konvolucijska_mapa = self.trazenje_znacajki(slika[:, :, 0], trenutni_filter)
                for broj_kanala in range(1, slika.shape[-1]):
                    konvolucijska_mapa = konvolucijska_mapa + self.trazenje_znacajki(
                        slika[:, :, broj_kanala], trenutni_filter)
            else:
                konvolucijska_mapa = self.trazenje_znacajki(slika, trenutni_filter)
            mapa_znacajki[:, :, broj_filtera] = konvolucijska_mapa
        return relu(mapa_znacajki)

    def generiraj_tezinske_faktore(self, ulazni_sloj):
        self.tf_ss = np.random.random((self.skriveni_sloj, ulazni_sloj))
        self.tf_is = np.random.random((self.izlazni_sloj, self.skriveni_sloj))
        self.o_ss = np.zeros((self.skriveni_sloj, 1))
        self.o_is = np.zeros((self.izlazni_sloj, 1))

    def krizna_entropija(self, predvidanje: np.ndarray, oznaka: np.ndarray, m: int) -> np.ndarray:
        logprobs = np.multiply(oznaka, np.log(predvidanje)) + np.multiply((1 - oznaka), np.log(1 - predvidanje))
        return - np.sum(logprobs) / m

    def mse(self, predvidanje: np.ndarray, oznaka: np.ndarray) -> np.ndarray:
        return np.square(predvidanje - oznaka).mean()

    def ucenje(self, broj_iteracija_konvolucije: int,
               broj_epoha: int,
               naziv_spremljenog_modela: str) -> None:
        popis_gubitaka: List[float] = []
        skup_za_ucenje: np.ndarray = self.podaci[:int(len(self.podaci) * .75)]
        broj_parametara: int = len(skup_za_ucenje)
        for _ in tqdm(range(broj_epoha)):
            for parametri, oznaka in tqdm(skup_za_ucenje):
                izravnati_niz: np.ndarray = self.konvolucijski_sloj(parametri, broj_iteracija_konvolucije)
                skriveni_sloj, izlazni_sloj, gubitak = self.guranje_naprijed(izravnati_niz, oznaka, broj_parametara)
                popis_gubitaka.append(gubitak)
                self.propagiranje_unazad(izravnati_niz, skriveni_sloj, izlazni_sloj, oznaka, broj_parametara)
        plt.plot(popis_gubitaka)
        plt.show()
        nauceni_tezinski_faktori_i_odstupanja = [self.tf_ss, self.tf_is, self.o_ss, self.o_is, broj_iteracija_konvolucije]
        self.spremljeni_parametri: str = naziv_spremljenog_modela + '.npy'
        np.save(self.spremljeni_parametri, nauceni_tezinski_faktori_i_odstupanja)


    def konvolucijski_sloj(self, parametri: np.ndarray,
                           broj_iteracija: int) -> np.ndarray:
        mape_znacajki: np.ndarray = None
        for _ in range(broj_iteracija):
            mape_znacajki = self.konvolucija(parametri if mape_znacajki is None else umanjene_mape, konvolucijski_filteri)
            umanjene_mape: np.ndarray = self.udruzivanje_slike(mape_znacajki)
        izravnati_niz = umanjene_mape.reshape(-1, 1)
        return izravnati_niz / np.linalg.norm(izravnati_niz)

    def guranje_naprijed(self, podaci: np.ndarray,
                         oznaka: np.ndarray,
                         velicina_skupa: int) -> Tuple[np.ndarray, np.ndarray, float]:
        if self.tf_ss is None:
            self.generiraj_tezinske_faktore(podaci.shape[0])
        skriveni_sloj: np.ndarray = relu(np.dot(self.tf_ss, podaci) + self.o_ss)
        izlazni_sloj: np.ndarray = softmax(np.dot(self.tf_is, skriveni_sloj) + self.o_is)
        gubitak: float = self.krizna_entropija(izlazni_sloj, oznaka, velicina_skupa)
        return skriveni_sloj, izlazni_sloj, gubitak

    def propagiranje_unazad(self, podaci: np.ndarray,
                             skriveni_sloj: np.ndarray,
                             izlazni_sloj: np.ndarray,
                             oznaka: np.ndarray,
                             broj_parametara: int) -> None:
        derivat_izlaznog_sloja: np.ndarray = izlazni_sloj - oznaka
        derivat_tf_izlaznog_sloja: np.ndarray = (1 / broj_parametara) * np.dot(derivat_izlaznog_sloja, skriveni_sloj.T)
        derivat_odstupanja_izlaznog_sloja: np.ndarray = (1 / broj_parametara) * np.sum(derivat_izlaznog_sloja, axis=1, keepdims=True)

        derivat_skrivenog_sloja: np.ndarray = np.multiply(np.dot(self.tf_is.T, derivat_izlaznog_sloja), 1 - np.power(skriveni_sloj, 2))
        derivat_tf_skrivenog_sloja: np.ndarray = (1 / broj_parametara) * np.dot(derivat_skrivenog_sloja, podaci.T)
        derivat_odstupanja_skrivenog_sloja: np.ndarray = (1 / broj_parametara) * np.sum(derivat_skrivenog_sloja, axis=1, keepdims=True)

        self.tf_ss = self.tf_ss - self.stopa_ucenja * derivat_tf_skrivenog_sloja
        self.o_ss = self.o_ss - self.stopa_ucenja * derivat_odstupanja_skrivenog_sloja
        self.tf_is = self.tf_is - self.stopa_ucenja * derivat_tf_izlaznog_sloja
        self.o_is = self.o_is - self.stopa_ucenja * derivat_odstupanja_izlaznog_sloja


    def testiranje(self) -> None:
        skup_za_testiranje: np.ndarray = self.podaci[int(len(self.podaci) * .75):]
        self.tf_ss, self.tf_is, self.o_ss, self.o_is = np.load('350_25_3k2.npy', allow_pickle=True)
        brojac: int = 0
        tocno: int = 0
        broj_parametara: int = len(skup_za_testiranje)
        for parametri, oznaka in tqdm(skup_za_testiranje):
            izravnati_niz: np.ndarray = self.konvolucijski_sloj(parametri, 2)
            _, izlazni_sloj, _ = self.guranje_naprijed(izravnati_niz, oznaka, broj_parametara)
            pozicija = np.where(oznaka == 1.)[0][0]
            if izlazni_sloj[pozicija] > .5:
                tocno += 1
            brojac += 1
        print('Točnost modela je: ' + str(round(tocno / brojac, 2) * 100) + '%')


    def test(self, skup_za_testiranje, spremljeni_parametri):
        self.tf_ss, self.tf_is, self.o_ss, self.o_is = np.load(spremljeni_parametri,
                                                                                           allow_pickle=True)
        brojac: int = 0
        tocno: int = 0
        broj_parametara: int = len(skup_za_testiranje)
        for parametri, oznaka in tqdm(skup_za_testiranje):
            izravnati_niz: np.ndarray = self.konvolucijski_sloj(parametri, 2)
            _, izlazni_sloj, _ = self.guranje_naprijed(izravnati_niz, oznaka, broj_parametara)
            pozicija = np.where(oznaka == 1.)[0][0]
            if izlazni_sloj[pozicija] > .5:
                tocno += 1
            brojac += 1
        print('Točnost modela je: ' + str(round(tocno / brojac, 2) * 100) + '%')