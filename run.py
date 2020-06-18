from izrada_podatkovnih_skupova import kreiraj_podatkovni_skup
from konvolucijska_neuronska_mreza import Konvolucijska_neuronska_mreza
from typing import List, Tuple
import numpy as np



def main():
    # podaci, oznaka = kreiraj_podatkovni_skup('100')
    # konv_mreza = Konvolucijska_neuronska_mreza(podaci, skriveni_sloj=20, izlazni_sloj=oznaka, stopa_ucenja=.008)
    # konv_mreza.feedforward()
    podaci, oznaka = kreiraj_podatkovni_skup('freya')
    test_mr = Konvolucijska_neuronska_mreza(podaci, skriveni_sloj=20, izlazni_sloj=oznaka, stopa_ucenja=.001)
    test_mr.test_ff()



if __name__ == '__main__':
    main()