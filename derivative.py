import numpy as np
from typing import Callable, Union
import torch

def deriv(func: Callable[[Union[np.ndarray, torch.tensor]], Union[np.ndarray, torch.tensor]],
          input_: Union[np.ndarray, torch.tensor],
          delta: float=0.001) -> Union[np.ndarray, torch.tensor]:
    '''
    To calculate the derivative of func(x) with respect to x
    :param func: original function
    :param input_: auto variable
    :param delta: derivative of auto variable
    :return: derivative of func
    '''
    return (func(input_ + delta) - func(input_)) / delta