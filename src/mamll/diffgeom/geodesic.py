import numpy as np
from scipy.optimize import minimize

from numpy.typing import NDArray
from typing import Callable

def _segment_energy(x0:NDArray[np.float64], x1:NDArray[np.float64], 
                    G:Callable[[NDArray[np.float64]], NDArray[np.float64]]) -> float:
    """Compute the energy of a segment between two points.
    
    Args:
    -------
        x0: NDArray[np.float64]
            The starting point of the segment.
        x1: NDArray[np.float64]
            The ending point of the segment.
        G: Callable[[NDArray[np.float64]], NDArray[np.float64]]
            A function that returns the metric tensor at a given point.
    
    Returns:
    --------
        energy: float
            The energy of the segment.
    """
    mid_point = (x0 + x1) / 2
    g = G(mid_point)
    segment_length = np.linalg.norm(x1 - x0)
    return segment_length * np.trace(g) / 2

def _path_energy(G: Callable[[NDArray[np.float64]], NDArray[np.float64]], x: NDArray[np.float64], n:int, start:NDArray[np.float64], end: NDArray[np.float64]) -> float:
    """Compute the energy of a piecewise linear path between two points.
    
    Args:
    -------
        G: Callable[[NDArray[np.float64]], NDArray[np.float64]]
            A function that returns the metric tensor at a given point.
        x: NDArray[np.float64]
            The sequence of points in the path.
        n: int
            The number of points in the path.
        start: NDArray[np.float64]
            The starting point.
        end: NDArray[np.float64]
            The ending point.
    
    Returns:
    --------
        energy: float
            The energy of the path.
    """
    points = np.concatenate(([start], x.reshape((n-2, 2)), [end]))
    energy = sum(_segment_energy(points[i], points[i+1], G) for i in range(n-1))
    return energy

def minimize_energy(G: Callable[[NDArray[np.float64]], NDArray[np.float64]], n:int, 
                    start: NDArray[np.float64], end: NDArray[np.float64]) -> NDArray[np.float64]:
    """Minimize the energy of a piecewise linear path between two points.
    
    Args:
    -------
        G: Callable[[NDArray[np.float64]], NDArray[np.float64]]
            A function that returns the metric tensor at a given point.
        n: int
            The number of points in the path.
        start: NDArray[np.float64]
            The starting point.
        end: NDArray[np.float64]
            The ending point.
    
    Returns:
    --------
        optimized_points: NDArray[np.float64]
            The sequence of points in the path.
    """
    # Initial guess for the points (linear interpolation between start and end)
    initial_guess = np.linspace(start, end, n)[1:-1].flatten()
    
    # Objective function
    objective = lambda x: _path_energy(G, x, n, start, end) # noqa
    
    # Minimization
    result = minimize(objective, initial_guess, method='L-BFGS-B')
    
    # Reshape result to obtain the sequence of points
    optimized_points = np.concatenate(([start], result.x.reshape((n-2, 2)), [end]))
    return optimized_points