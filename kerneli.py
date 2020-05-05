import numpy as np

kernel = np.array(
    [[1., 0., 1.],
     [0., 1., 0.],
     [1., 0., 1.]]
)

konvolucijski_kernel = np.array(
    [[-1., -1., -1.],
     [-1., 8., -1.],
     [-1., -1., -1.]]
)

horizontalni_kernel = np.array(
    [[-1., -1., -1.],
     [0., 0., 0.],
     [1., 1., 1.]]
)

vertikalni_kernel = np.array(
    [[-1., 0., 1.],
     [-1., 0., 1.],
     [-1., 0., 1.]]
)