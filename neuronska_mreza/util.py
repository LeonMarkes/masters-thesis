import numpy as np
from typing import List


def skalarni_produkt(xs: List[float], ys: List[float]) -> float:
    return np.dot(xs, ys)


def sigmoidna_funkcija(x: float) -> float:
    return 1. / (1. + np.exp(-x))


def derivat_sigmoidne_funkcije(x: float) -> float:
    sig: float = sigmoidna_funkcija(x)
    return sig * (1 - sig)


def relu(x: float) -> float:
    return max(0, x > 0)


def swish_relu(x: float) -> float:
    return x * sigmoidna_funkcija(x)


def normalizacija_skaliranih_znacajki(dataset: List[List[float]]) -> None:
    for broj_stupca in range(len(dataset[0])):
        stupac: List[float] = [red[broj_stupca] for red in dataset]
        maksimum = max(stupac)
        minimum = min(stupac)
        for broj_retka in range(len(dataset)):
            dataset[broj_retka][broj_stupca] = (dataset[broj_retka][broj_stupca] - minimum) / (maksimum - minimum)
