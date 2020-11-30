# Klasifikacija slika uz pomoć konvolucijske neuronske mreže

Kod koji se primjenjivao u praktičnom dijelu rada se nalazi unutar skripti:
- konvolucijska_neuronska_mreza.py
- izrada_podatkovnih_skupova.py
- run.py
- util.py
- kerneli.py
- config.yml

Potrebni dodatni python library-ji:
```
pip install opencv-python
pip install numpy
pip install matplotlib
pip install tqdm
```
Osim njih još se koriste moduli:
```
yaml
typing
```

Rad mreže se pokreće s run.py skriptom.
Prvo se poziva kreiraj_podatkovni_skup('obrana') koja kao argument prima folder sa slikama mačaka i pasa. Tu vrijednost možete namjestiti u config.yml.
Slike su dohvaćene s ovoga linka: https://www.kaggle.com/chetankv/dogs-cats-images
Nakon toga se instancira objekt, koji kao argumente prima:
 - podatke (dobiveni preko kreiraj_podatkovni_skup), 
 - proizvoljan broj skrivenih slojeva (integer), 
 - oznake za podatke (dobiveni preko kreiraj_podatkovni_skup) te 
 - proizvoljnu stopu učenja (float).
 
Nakon toga se pokreće metoda ucenje() koja kao argumente prima:
- broj iteracija konvolucije (integer)
- broj epoha (integer)
- naziv dokumenta pod kojim se sprema naučeni model (string)

Ako se samo želi testirati već od prije naučeni model, potrebno je zakomentirati metodu učenje()

Metoda testiranje zaprima naziv naučenog modela te nakon što klasificira podatke, ispiše rezultat u konzolu.
Rad modela traje neko vrijeme (za 75 slika je potrebno otprilike 10 minuta rada).

U util.py se nalaze aktivacijske funkcije.
U kerneli.py se nalaze filteri.

izrada_podatkovnih_skupova.py radi pred procesiranje podataka. Promjeni veličinu slika u 350x350 piksela te kreira oznake za svaku sliku.
Da bi predobrada dobro radila, svaka slika mora ima naziv: ime_životinje.neki_redni_broj

Config.yml sadrži svu konfiguraciju potrebnu za rad modela:
- path_to_data -> putanja do datoteke gdje se nalaze slike, relativna na root datoteku
- skriveni_sloj -> broj neurona u skrivenom sloju (integer)
- stopa_ucenja -> stopa učenja (float)
- naziv_modela -> naziv modela s ekstenzijom .npy (string)
- broj_iteracija_konvolucije -> broj konvolucijskih slojeva (integer)
- broj_epoha -> broj epoha (integer)

konvolucijska_neuronska_mreza.py sadrži sav kod koji se koristi pri klasifikaciji i obradi podataka.
hijerarhija koda je sljedeća:

![Alt text](graf.png?raw=true "Struktura koda")
