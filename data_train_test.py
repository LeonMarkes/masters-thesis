# from neuronska_mreza.neuronska_mreza import Neuronska_mreza
from konvolucijska_neuronska_mreza import Konvolucijska_neuronska_mreza
import numpy as np
import os
from typing import List
import csv



def main():
    popis_datoteka: List[str] = ['cats', 'dogs']
    parametri_zivotinja: List[List[float]] = []
    klasifikacija_zivotinja: List[List[float]] = []
    zivotinje: List[List[float]] = []
    for naziv_datoteke in popis_datoteka:
        for zivotinja in os.listdir(naziv_datoteke + '/'):
            konvolucija: Konvolucijska_neuronska_mreza = Konvolucijska_neuronska_mreza(
                naziv_datoteke + '/' + zivotinja)
            konvolucija.pozivanje_konvolucijskog_sloja(3)
            svojstva_zivotinja: List[float] = konvolucija.zgusnjavanje_izlaza()
            if naziv_datoteke == 'dogs':
                svojstva_zivotinja.append(1.)
                zivotinje.append(svojstva_zivotinja)
            if naziv_datoteke == 'cats':
                svojstva_zivotinja.append(0.)
                zivotinje.append(svojstva_zivotinja)
        print(naziv_datoteke)

    with open('zivotinje.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(zivotinje)

    # with open('zivotinje.txt', 'a+') as file:
    #     for red in zivotinje:
    #         file.write(str(red) + '\n')






    # neuronska_mreza: Neuronska_mreza = Neuronska_mreza([1024, 8, 2], 0.5)
    # ne nalazi module

    # dohvati folder sa slikama
    # kreiraj dvije liste
    # dohvati podatke sa slika
    # lista sa svojstvima
    # lista s oznakama
    # koristi regex za dohvaćanje koja je životinja
    # te train test split
    # i onda učenje i validacija
    return


if __name__ == '__main__':
    main()
