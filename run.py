from izrada_podatkovnih_skupova import kreiraj_podatkovni_skup
from konvolucijska_neuronska_mreza import Konvolucijska_neuronska_mreza
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt



def main():
    podaci, oznaka = kreiraj_podatkovni_skup('obrana') # new2
    konv_mreza = Konvolucijska_neuronska_mreza(podaci, skriveni_sloj=25, izlazni_sloj=oznaka, stopa_ucenja=.001)
    # konv_mreza.ucenje(3, 3, '350_25_3k')
    konv_mreza.testiranje()


if __name__ == '__main__':
    main()