from typing import Callable, Union
import numpy as np
from numpy.typing import NDArray
import torch

def discrete_length(curve: Union[Callable[[NDArray[np.float64]], NDArray[np.float64]],
                                 Callable[[torch.Tensor], torch.Tensor]]
                                 , t: Union[NDArray[np.float64], torch.Tensor]) -> Union[float, torch.Tensor]:
    """Compute the length of a discrete curve.

    Supports both numpy arrays and pytorch tensors.
    
    Args:
    ----------
        curve (Union[Callable[[NDArray[np.float64]], NDArray[np.float64]], Callable[[torch.Tensor], torch.Tensor]]): A function that takes a parameter `t` and returns a point on the curve.
        t (Union[NDArray[np.float64], torch.Tensor]): The parameter values to sample the curve at.
        
    Returns:
    ----------
        Union[float, torch.Tensor]: The length of the discrete curve.
    """
    # Approximate the length of the curve by summing the distances between consecutive points
    # points are in column vector
    ct = curve(t) # type: ignore
    return np.sum(np.linalg.norm(ct[:, :-1] - ct[:, 1:], axis=0), axis=0) # type: ignore

def analytical_continuous_length(dcurve: Callable[[float], NDArray[np.float64]], a: float, b: float, n: int) -> float:
    """Compute the length of a continuous curve.
    
    Args:
    ----------
        dcurve (Callable[[float], NDArray[np.float64]]): Derivative of the curve function that takes a parameter `t` and returns a point on the curve.
        a (float): The start of the interval.
        b (float): The end of the interval.
        n (int): The number of points to sample.
        
    Returns:
    ----------
        float: The length of the continuous curve.
    """
    # compute the length of the curve by approximating the integral
    t = np.linspace(a + np.finfo(float).eps, b - np.finfo(float).eps, n)
    return np.sum(np.linalg.norm(dcurve(t), axis=1)) * (b - a) / n # type: ignore