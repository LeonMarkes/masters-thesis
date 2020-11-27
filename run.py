from izrada_podatkovnih_skupova import kreiraj_podatkovni_skup
from konvolucijska_neuronska_mreza import Konvolucijska_neuronska_mreza
import yaml


def main():
    # edit configuration in config.yml file
    with open('config.yml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    podaci, oznaka = kreiraj_podatkovni_skup(config['path_to_data'])
    konv_mreza = Konvolucijska_neuronska_mreza(podaci, skriveni_sloj=config['skriveni_sloj'], izlazni_sloj=oznaka, stopa_ucenja=config['stopa_ucenja'], spremljeni_parametri=config['naziv_modela'])
    konv_mreza.ucenje(config['broj_iteracija_konvolucije'], config['broj_epoha']) # ako želite samo testirati model postojeći model, komentirajte ovu liniju koda
    konv_mreza.testiranje()


if __name__ == '__main__':
    main()