from typing import List, Callable, Optional
from random import random
from neuron import Neuron
from util import skalarni_produkt


class Sloj:
    def __init__(self,
                 prethodni_sloj: Optional,
                 broj_neurona: int,
                 stopa_ucenja: float,
                 aktivacijska_funkcija: Callable[[float], float],
                 derivat_aktivacijske_funkcije: Callable[[float], float]) -> None:
        self.prethodni_sloj: Optional[Sloj] = prethodni_sloj
        self.neuroni: List[Neuron] = []
        for _ in range(broj_neurona):
            if prethodni_sloj is None:
                slucajni_uteg: List[float] = []
            else:
                slucajni_uteg = [random() for _ in range(len(prethodni_sloj.neuroni))]
            neuron: Neuron = Neuron(slucajni_uteg, stopa_ucenja, aktivacijska_funkcija, derivat_aktivacijske_funkcije)
            self.neuroni.append(neuron)

        self.izlazna_memorija: List[float] = [.0 for _ in range(broj_neurona)]


    def izlazne_vrijednosti(self, ulazne_vrijednosti: List[float]) -> List[float]:
        if self.prethodni_sloj is None:
            self.izlazna_memorija = ulazne_vrijednosti
        else:
            self.izlazna_memorija = [n.izlazna_vrijednost(ulazne_vrijednosti) for n in self.neuroni]
        return self.izlazna_memorija


    def izracun_delte_za_izlazni_sloj(self, ocekivanja: List[float]) -> None:
        for n in range(len(self.neuroni)):
            self.neuroni[n].delta = self.neuroni[n].derivat_aktivacijske_funkcije(
                self.neuroni[n].izlazna_memorija) * (ocekivanja[n] - self.izlazna_memorija[n])


    def izracun_delte_za_skriveni_sloj(self, sljedeci_sloj) -> None:
        for indeks, neuron in enumerate(self.neuroni):
            sljedeci_uteg: List[float] = [n.utezi[indeks] for n in sljedeci_sloj.neuroni]
            sljedeca_delta: List[float] = [n.delta for n in sljedeci_sloj.neuroni]
            suma_utega_i_delti: float = skalarni_produkt(sljedeci_uteg, sljedeca_delta)
            neuron.delta = neuron.derivat_aktivacijske_funkcije(neuron.izlazna_memorija) * suma_utega_i_delti

