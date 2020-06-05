import numpy as np

kernel = np.array(
    [[1., 0., 1.],
     [0., 1., 0.],
     [1., 0., 1.]]
)

# detekcija_ruba = np.array(
#     [[-1., -1., -1.],
#      [-1., 8., -1.],
#      [-1., -1., -1.]]
# )
#
# horizontalni_kernel = np.array(
#     [[-1., -1., -1.],
#      [0., 0., 0.],
#      [1., 1., 1.]]
# )
#
# vertikalni_kernel = np.array(
#     [[-1., 0., 1.],
#      [-1., 0., 1.],
#      [-1., 0., 1.]]
# )

konvolucijski_filteri = np.zeros((2, 3, 3))

konvolucijski_filteri[0, :, :] = np.array([[[-1, 0, 1],
                                            [-1, 0, 1],
                                            [-1, 0, 1]]])

konvolucijski_filteri[1, :, :] = np.array([[[1, 1, 1],
                                            [0, 0, 0],
                                            [-1, -1, -1]]])

detekcija_ruba = np.zeros((1, 3, 3))

detekcija_ruba[0, :, :] = np.array([[[-1, -1, -1],
                                     [-1, 8, -1],
                                     [-1, -1, -1]]])
