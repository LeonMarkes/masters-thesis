import numpy as np
from typing import List


def skalarni_produkt(xs: List[float], ys: List[float]) -> float:
    return np.dot(xs, ys)


def sigmoidna_funkcija(x: float) -> float:
    return 1. / (1. + np.exp(-x))


def derivat_sigmoidne_funkcije(x: float) -> float:
    sig: float = sigmoidna_funkcija(x)
    return sig * (1 - sig)


def normalize_by_feature_scaling(dataset: List[List[float]]) -> None:
    for col_num in range(len(dataset[0])):
        column: List[float] = [row[col_num] for row in dataset]
        maximum = max(column)
        minimum = min(column)
        for row_num in range(len(dataset)):
            dataset[row_num][col_num] = (dataset[row_num][col_num] - minimum) / (maximum - minimum)
