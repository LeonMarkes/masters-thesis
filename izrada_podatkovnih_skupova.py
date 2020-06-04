'''
napiši program s kojim ćeš dohvatiti slike
kreirati podatkovni skup za učenje
crno bijele slike istih dimenzija
kreiraj drugi skup s oznakama zivotinja
'''
from typing import List, Tuple
from multiprocessing import Process
import numpy as np
import os
import cv2


def oznaci_sliku(naziv_datoteke: str) -> Tuple[List[float], int]:  # prolazi kroz datoteku i kreira matricu identiteta čija
    #  veličina ovisi o broju jedinstevih slika/životinja u njoj
    jedinstvene_vrijednosti: List[str] = []
    for slika in os.listdir(naziv_datoteke):  # provjeri jedinstven broj životinja
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
        print(vrijednost + ' je označena s ' + str(oznake[jedinstvene_vrijednosti.index(vrijednost)]))

    return oznake_podataka, len(jedinstvene_vrijednosti)


def dohvati_i_uredi_slike(naziv_datoteke: str,
                          velicina_slike: Tuple[int] = (200, 200)) -> List[float]:
    slike: List[List[float]] = []
    putanja_do_datoteke = os.getcwd() + '\\' + naziv_datoteke
    for naziv_slike in os.listdir(naziv_datoteke):
        putanja = os.path.join(putanja_do_datoteke, naziv_slike)
        slika = cv2.resize(cv2.imread(putanja, cv2.IMREAD_GRAYSCALE), velicina_slike)
        slike.append(slika)
    return slike


def kreiraj_podatkovni_skup(naziv_datoteke: str = 'podatkovni_skup') -> Tuple[List[np.ndarray], int]:
    podatkovni_skup: List[List[float]] = []
    oznake: Tuple[List[float], int] = oznaci_sliku(naziv_datoteke)
    slike = dohvati_i_uredi_slike(naziv_datoteke)
    for slika, oznaka in zip(slike, oznake[0]):
        podatkovni_skup.append([np.array(slika), np.array(oznaka)])
    np.random.shuffle(podatkovni_skup)
    return podatkovni_skup, oznake[1]

