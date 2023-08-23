import numpy as np


def harmonic_potentia_1d(x, k=1):
    """
    Harmonic potential
    """
    return 1/2*k*x**2


def muller_potential(vec):
    """
    Muller Brown potential
    """ 
    A = np.array([-200.0, -100.0, -170.0, 15])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    potential = 0
    x, y = vec[0], vec[1]
    for j in range(4):
        x_x0 = x - x0[j]
        y_y0 = y - y0[j]
        potential += A[j]*np.exp(a[j]*x_x0**2 + b[j]*x_x0*y_y0 + c[j]*y_y0**2)
    return 0.05*potential


def potential_to_probability(f, kbT=2.479):
    """
    Convert potential to probability
    """
    def wrapped_function(*args, **kwargs):
        return np.exp(-f(*args, **kwargs) / kbT)
    return wrapped_function


def MCMC(f, num_samples, step_size, x0, xrange):
    """
    Metropolis-Hastings algorithm
    """
    x = np.zeros(num_samples)
    y = np.zeros(num_samples)
    samples = np.zeros((num_samples, len(x0)))
    samples[0]= x0
    for i in range(num_samples - 1):
        x_cand = samples[i].copy()
        for j in range(len(x0)):
            x_cand[j] = samples[i][j] + np.random.normal(0, step_size)
        in_range = np.all(np.logical_and(x_cand > xrange[:, 0], x_cand < xrange[:, 1]))
        while np.random.rand() > f(x_cand) / f(samples[i]) or not in_range:
            x_cand = samples[i].copy()
            for j in range(len(x0)):
                x_cand[j] = samples[i][j] + np.random.normal(0, step_size)
            in_range = np.all(np.logical_and(x_cand > xrange[:, 0], x_cand < xrange[:, 1]))
        samples[i + 1] = x_cand
    return samples