import numpy as np
from math import erf
from scipy.stats import poisson

types = ['normal', 'cauchy', 'laplace', 'poisson', 'uniform']


def get_distribution(name, size):
    match name:
        case 'normal':
            return np.random.normal(0, 1, size)
        case 'cauchy':
            return np.random.standard_cauchy(size)
        case 'laplace':
            return np.random.laplace(0, 1/np.sqrt(2), size)
        case 'poisson':
            return np.random.poisson(10, size)
        case 'uniform':
            return np.random.uniform(-np.sqrt(3), np.sqrt(3), size)
    return []


def get_density(name, array):
    match name:
        case 'normal':
            return [1 / (np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2) for x in array]
        case 'cauchy':
            return [1 / (np.pi * (x**2 + 1)) for x in array]
        case 'laplace':
            return [1 / np.sqrt(2) * np.exp(-np.sqrt(2) * np.fabs(x)) for x in array]
        case 'poisson':
            return [10 ** float(x) * np.exp(-10) / np.math.gamma(x) for x in array]
        case 'uniform':
            return [1 / (2 * np.sqrt(3)) if abs(x) <= np.sqrt(3) else 0 for x in array]
    return []


def get_func(name, x):
    match name:
        case 'normal':
            return 0.5 * (1 + erf(x / np.sqrt(2)))
        case 'cauchy':
            return np.arctan(x) / np.pi + 0.5
        case 'laplace':
            if x <= 0:
                return 0.5 * np.exp(np.sqrt(2) * x)
            else:
                return 1 - 0.5 * np.exp(-np.sqrt(2) * x)
        case 'poisson':
            return poisson.cdf(x, 10)
        case 'uniform':
            if x < -np.sqrt(3):
                return 0
            elif np.fabs(x) <= np.sqrt(3):
                return (x + np.sqrt(3)) / (2 * np.sqrt(3))
            else:
                return 1
    return None


def compute_trimmed_mean(array):
    r = int(len(array) / 4)
    sorted_array = np.sort(array)
    sum = 0.0
    for i in range(r + 1, len(array) - r):
        sum += sorted_array[i]
    return sum / (len(array) - 2 * r)

