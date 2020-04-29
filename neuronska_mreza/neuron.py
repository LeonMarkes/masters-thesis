from typing import List, Callable
from util import skalarni_produkt


class Neuron:

    def __init__(self,
                 tezinski_faktori: List[float],
                 stopa_ucenja: float,
                 aktivacijska_funkcija: Callable[[float], float],
                 derivat_aktivacijske_funkcije: Callable[[float], float]) -> None:
        self.tezinski_faktori: List[float] = tezinski_faktori
        self.aktivacijska_funkcija: Callable[[float], float] = aktivacijska_funkcija
        self.derivat_aktivacijske_funkcije: Callable[[float], float] = derivat_aktivacijske_funkcije
        self.stopa_ucenja: float = stopa_ucenja
        self.izlazna_memorija: float = 0.
        self.delta: float = 0.

    def izlazna_vrijednost(self, ulazne_vrijednosti: List[float]) -> float:
        self.izlazna_memorija = skalarni_produkt(ulazne_vrijednosti, self.tezinski_faktori)
        return self.aktivacijska_funkcija(self.izlazna_memorija)
