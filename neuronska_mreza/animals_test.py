import csv
from typing import List
from util import normalizacija_skaliranih_znacajki
from neuronska_mreza.neuronska_mreza import Neuronska_mreza
from random import shuffle
import numpy as np
import time

if __name__ == '__main__':
    start_time = time.time()
    star_parameters: List[List[float]] = []
    star_classifications: List[List[float]] = []
    star_types: List[int] = []
    with open('zivotinje.csv', mode='r') as star_file:
        stars: List = list(csv.reader(star_file, quoting=csv.QUOTE_NONNUMERIC))
        shuffle(stars)
        for star in stars:
            parameters: List[float] = [float(n) for n in star[:-1]]
            star_parameters.append(parameters)
            types: float = star[-1]
            if types == 1.:
                star_classifications.append([1., 0.])
            elif types == 0.:
                star_classifications.append([0., 1.])
            star_types.append(types)

        normalizacija_skaliranih_znacajki(star_parameters)

        star_network: Neuronska_mreza = Neuronska_mreza([1024, 15, 2], 0.6)

        def star_interpreter_output(output: List[float]) -> int:
            if max(output) == output[0]:
                return 1
            else:
                return 0


        star_trainers: List[List[float]] = star_parameters[:100]
        star_trainers_corrects: List[List[float]] = star_classifications[:100]
        for _ in range(10):
            star_network.ucenje(star_trainers, star_trainers_corrects)

        star_testers: List[List[float]] = star_parameters[101:]
        star_testers_corrects: List[int] = star_types[101:]
        star_results = star_network.validacija(star_testers, star_testers_corrects, star_interpreter_output)
        print(f"{star_results[0]} correct of {star_results[1]} = {star_results[2] * 100}%")
        print('--- %s seconds ---' % (round(time.time() - start_time, 3)))
