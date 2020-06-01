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

def oznaci_sliku(naziv_datoteke: str) -> List[float]:  #  prolazi kroz datoteku i kreira matricu identiteta čija
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

    return oznake_podataka


def dohvati_i_uredi_slike(naziv_datoteke: str,
                           velicina_slike: Tuple[int] = (200, 200)) -> List[float]:
    slike: List[List[float]] = []
    os.getcwd()
    for naziv_slike in os.listdir(naziv_datoteke):
        slika:




def kreiraj_skup_za_ucenje(naziv_datoteke: str = 'skup_za_ucenje') -> None:
    skup_za_ucenje: List[List[float]] = []
    oznake = oznaci_sliku(naziv_datoteke)
    slike = dohvati_i_uredi_slike(naziv_datoteke)


    # for slika in os.listdir(naziv_datoteke):
    #     oznaka: List[float] = oznaci_sliku(slika)
    # učitaj slike iz datoteka
    # resizeaj ih da sve budu iste
    # kreiraj numpy array s svojstvima i klasama np.eye

    return


def main():
    kreiraj_skup_za_ucenje()


if __name__ == '__main__':
    main()
