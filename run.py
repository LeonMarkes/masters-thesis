from izrada_podatkovnih_skupova import kreiraj_podatkovni_skup
from konvolucijska_neuronska_mreza import Konvolucijska_neuronska_mreza
from typing import List, Tuple
import numpy as np



def main():
    podaci: Tuple[List[np.ndarray], int] = kreiraj_podatkovni_skup()
    print(podaci[0])
    konv_mreza = Konvolucijska_neuronska_mreza(podaci[0], skriveni_sloj=10, izlazni_sloj=podaci[1])
    konv_mreza.treniranje(500)


if __name__ == '__main__':
    main()