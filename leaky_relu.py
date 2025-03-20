import numpy as np

def leaky_relu(x: np.ndarray) -> np.ndarray:
    ''' To calculate the leaky relu w.r.t. x
    :param x: x
    :return: max(eps*x, x)
    '''
    eps = 1e-3
    return np.maximum(eps*x, x)