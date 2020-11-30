'''
napiši program s kojim ćeš dohvatiti slike
kreirati podatkovni skup za učenje
crno bijele slike istih dimenzija
kreiraj drugi skup s oznakama zivotinja
'''
from typing import List, Tuple
import numpy as np
import os
import cv2


def oznaci_sliku(naziv_datoteke: str) -> Tuple[List[float], int]:
    '''
    Ovisno o broju različitih naziva slika, kreirat će odgovarajući broj oznaka.
    '''
    jedinstvene_vrijednosti: List[str] = []
    for slika in os.listdir(naziv_datoteke):
        naziv_slike: str = slika.split('.')[0]
        if naziv_slike not in jedinstvene_vrijednosti:
            jedinstvene_vrijednosti.append(naziv_slike)
    oznake_podataka: List[List[float]] = []
    oznake: List[List[float]] = np.eye(len(jedinstvene_vrijednosti)).tolist()
    for slika in os.listdir(naziv_datoteke):
        naziv_slike: str = slika.split('.')[0]
        pozicija: int = jedinstvene_vrijednosti.index(naziv_slike)
        oznake_podataka.append(oznake[pozicija])
    for vrijednost in jedinstvene_vrijednosti:
        print(vrijednost + ' je označen s ' + str(oznake[jedinstvene_vrijednosti.index(vrijednost)]))
    return oznake_podataka, len(jedinstvene_vrijednosti)


def dohvati_i_uredi_slike(naziv_datoteke: str,
                          velicina_slike: Tuple[int] = (350, 350)) -> List[float]:
    '''
    Svim slikama iz datoteke promijeni dimenzije i pretvori ih u crno bijele slike.
    '''
    slike: List[List[float]] = []
    # dohvati sve slike iz datoteke, te im promjeni dimenzije i
    # sve slike u boji pretvori u crno bijele slike
    for naziv_slike in os.listdir(naziv_datoteke):
        putanja = os.path.join(naziv_datoteke, naziv_slike)
        slika = cv2.resize(cv2.imread(putanja, cv2.IMREAD_GRAYSCALE), velicina_slike)
        slike.append(slika)
    return slike


def kreiraj_podatkovni_skup(naziv_datoteke: str = 'podatkovni_skup') -> Tuple[List[np.ndarray], int]:
    '''
    Funkcija pomoću oznaci_sliku() i dohvati_i_uredi_sliku()
    kreira uređeni podatkovni skup koji se prosljeđuje modelu.
    '''
    podatkovni_skup: List[List[float]] = []
    oznake: Tuple[List[float], int] = oznaci_sliku(naziv_datoteke)
    slike = dohvati_i_uredi_slike(naziv_datoteke)
    for slika, oznaka in zip(slike, oznake[0]):
        podatkovni_skup.append([np.array(slika), np.array(oznaka).reshape(-1, 1)])
    np.random.shuffle(podatkovni_skup)
    np.save('podatkovni_skup.npy', podatkovni_skup)
    return podatkovni_skup, oznake[1]

