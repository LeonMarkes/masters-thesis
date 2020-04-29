import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-8, 8, .1)
f = 1 / (1 + np.exp(-x))

plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()


