from izrada_podatkovnih_skupova import kreiraj_podatkovni_skup
from konvolucijska_neuronska_mreza import Konvolucijska_neuronska_mreza


def main():
    podaci, oznaka = kreiraj_podatkovni_skup('obrana')
    konv_mreza = Konvolucijska_neuronska_mreza(podaci, skriveni_sloj=25, izlazni_sloj=oznaka, stopa_ucenja=.001, spremljeni_parametri='350_25_3k2.npy')
    # konv_mreza.ucenje(3, 3) # ako želite trenirati novi model, odkomentirajte ovu liniju koda
    konv_mreza.testiranje()


if __name__ == '__main__':
    main()