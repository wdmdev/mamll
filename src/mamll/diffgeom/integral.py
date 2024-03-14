import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple

def integral_2d(f: Callable[[float, float], float], 
                x1_range: Tuple[float, float], x2_range:Tuple[float, float], 
                N:int=1000) -> float:
    """
    Calculate the double integral of f over specified ranges using the trapezoidal rule.
    
    Args:
    --------
    f: The function to integrate. It must be a function of two variables, f(x1, x2).
    x1_range: A tuple (min, max) specifying the range of x1.
    x2_range: A tuple (min, max) specifying the range of x2.
    N: The number of points to use in each dimension (default 1000).
    
    Returns:
    --------
    - The approximate value of the double integral of f over the given ranges.
    """
    x1_min, x1_max = x1_range
    x2_min, x2_max = x2_range
    
    # Generate linearly spaced points for x1 and x2
    x1 = np.linspace(x1_min, x1_max, N)
    x2 = np.linspace(x2_min, x2_max, N)
    
    # Differential elements
    dx1 = (x1_max - x1_min) / (N - 1)
    dx2 = (x2_max - x2_min) / (N - 1)
    
    # Calculate the integral
    integral = 0
    for i in range(N):
        for j in range(N):
            integral += f(x1[i], x2[j]) * dx1 * dx2
    
    return integral


def integral_2d_vectorized(f: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], 
                           x1_range: Tuple[float, float], 
                           x2_range: Tuple[float, float], 
                           N: int = 1000) -> float:
    """
    Calculate the double integral of f over specified ranges using a vectorized approach.
    
    Args:
    --------
    f: The function to integrate. It must be a function that can accept numpy arrays for x1 and x2.
    x1_range: A tuple (min, max) specifying the range of x1.
    x2_range: A tuple (min, max) specifying the range of x2.
    N: The number of points to use in each dimension (default 1000).
    
    Returns:
    --------
    - The approximate value of the double integral of f over the given ranges.
    """
    x1_min, x1_max = x1_range
    x2_min, x2_max = x2_range
    
    # Generate linearly spaced points for x1 and x2
    x1 = np.linspace(x1_min, x1_max, N)
    x2 = np.linspace(x2_min, x2_max, N)
    
    # Create meshgrid
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    
    # Differential elements
    dx1 = (x1_max - x1_min) / (N - 1)
    dx2 = (x2_max - x2_min) / (N - 1)
    
    # Evaluate the function on the grid and calculate the integral using vectorization
    integral = np.sum(f(X1, X2)) * dx1 * dx2
    
    return integral # type: ignore