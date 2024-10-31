import sympy as sp
import numpy as np
import random

def eval_sympy(objective_function, x):
    """
    Parameters:
    objective_function (sympy exp): Objective function, which can be a sympy expression or a Python callable
    x (np.ndarray):  Array of values to substitute into the objective function

    Returns:
    - f(x) (float): The evaluated function value
    """
    
    if isinstance(objective_function, sp.Expr):
        sorted_symbols = sorted(objective_function.free_symbols, key=lambda s: s.name)
        subs_dict = {symbol: value for symbol, value in zip(sorted_symbols, x.tolist())}
        result = objective_function.subs(subs_dict)
    else:
        result = objective_function(*x)

    return float(result)


def spread_factor(u=None, nc=2):
    """
    Computes the spread factor for Simulated Binary Crossover (SBX) given the u value

    Parameters:
    - u (float): random u value from 0 to 1
    - nc (int): n_c value, n=0 uniform distribution, 2<n<5 matches closely the simulation for single-point crossover

    Returns:
    - beta (float)
    """
    if u is None:
        u = random.random()

    if u <= 0.5:
        beta = (2 * u) ** (1/(nc + 1))
    else:
        beta = (1/(2 * (1 - u))) ** (1/(nc + 1))
    return beta

def beta_q_factor(delta, eta_m, u=None):
    """
    Computes the beta_q factor for parameter based mutation (PM)

    Parameters:
    - delta (float): value calculated with y parent solution, y upper and lower limits.
    - eta_m (float): 100 + generation number aka η_m
    
    Returns:
    - Returns:
    - beta_q (float)
    """    
    if u is None:
        u = np.random.rand(*delta.shape)
    
    delta_q = np.where(
        u <= 0.5,
        (2 * u + (1 - 2 * u) * (1 - delta) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1,
        1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta) ** (eta_m + 1)) ** (1 / (eta_m + 1))
    )

    return delta_q

