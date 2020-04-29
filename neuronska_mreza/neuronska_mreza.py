from typing import List, Callable, TypeVar, Tuple
from functools import reduce
from sloj import Sloj
from util import sigmoidna_funkcija, derivat_sigmoidne_funkcije

T = TypeVar('T')


class Neuronska_mreza:
    def __init__(self,
                 struktura_slojeva: List[int],
                 stopa_ucenja: float,
                 aktivacijska_funkcija: Callable[[float], float] = sigmoidna_funkcija,
                 derivat_aktivacijske_funkcije: Callable[[float], float] = derivat_sigmoidne_funkcije) -> None:
        if len(struktura_slojeva) < 3:
            raise ValueError('Error: potrebno je minimalno 3 sloja. Ulazni, skriveni i izlazni')
        self.slojevi: List[Sloj] = []

        ulazni_sloj: Sloj = Sloj(None, struktura_slojeva[0], stopa_ucenja, aktivacijska_funkcija, derivat_aktivacijske_funkcije)
        self.slojevi.append(ulazni_sloj)

        for prethodni, broj_neurona in enumerate(struktura_slojeva[1::]):
            sljedeci_sloj: Sloj = Sloj(self.slojevi[prethodni], broj_neurona, stopa_ucenja, aktivacijska_funkcija,
                                       derivat_aktivacijske_funkcije)
            self.slojevi.append(sljedeci_sloj)

    def izlazne_vrijednosti(self, ulazna_vrijednost: List[float]) -> List[float]:
        return reduce(lambda ulazne_vrijednosti, sloj: sloj.izlazne_vrijednosti(ulazne_vrijednosti), self.slojevi, ulazna_vrijednost)

    def propagiranje_unatraske(self, ocekivanje: List[float]) -> None:
        posljednji_sloj: int = len(self.slojevi) - 1
        self.slojevi[posljednji_sloj].izracun_delte_za_izlazni_sloj(ocekivanje)
        for s in range(posljednji_sloj - 1, 0, -1):
            self.slojevi[s].izracun_delte_za_skriveni_sloj(self.slojevi[s + 1])

    def azuriraj_utege(self) -> None:
        for sloj in self.slojevi[1:]:
            for neuron in sloj.neuroni:
                for uteg in range(len(neuron.utezi)):
                    neuron.utezi[uteg] = neuron.utezi[uteg] + (neuron.stopa_ucenja * (sloj.prethodni_sloj.izlazna_memorija[uteg]) * neuron.delta)

    def ucenje(self, ulazne_vrijednosti: List[List[float]],
               ocekivanja: List[List[float]]) -> None:
        for polozaj, xs in enumerate(ulazne_vrijednosti):
            ys: List[float] = ocekivanja[polozaj]
            izlazi: List[float] = self.izlazne_vrijednosti(xs)
            self.propagiranje_unatraske(ys)
            self.azuriraj_utege()

    def validacija(self, ulazne_vrijednosti: List[List[float]],
                   ocekivanja: List[T],
                   tumacenje_izlaznih_vrijednosti: Callable[[List[float]], T]) -> Tuple[int, int, float]:
        ispravnost: int = 0
        for ulaz, ocekivanje in zip(ulazne_vrijednosti, ocekivanja):
            rezultat: T = tumacenje_izlaznih_vrijednosti(self.izlazne_vrijednosti(ulaz))
            if rezultat == ocekivanje:
                ispravnost += 1
        postotak: float = ispravnost / len(ulazne_vrijednosti)
        return ispravnost, len(ulazne_vrijednosti), postotak
