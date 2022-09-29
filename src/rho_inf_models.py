from os.path import join

import cluster_toolkit as ct
import numpy as np
from classy import Class
from numba import njit
from scipy.interpolate import interp1d

from src.config import COSMO_CLASS, RHOM, SDD

masses = np.load(join(SDD, "rho_model/rho_orb/mass_corrected.npy"))

# Instantiate the cosmology
classcosmo = Class()
classcosmo.set(COSMO_CLASS)
classcosmo.compute()

# Power spectrum
z = 0
kk = np.logspace(-6, 3, num=1_000)  # 1/Mpc
Plin = np.array([classcosmo.pk_lin(ki, z) for ki in kk])
Pnl = np.array([classcosmo.pk(ki, z) for ki in kk])
# NOTE: You will need to convert these to h/Mpc and (Mpc/h)^3
# to use in the toolkit. To do this you would do:
kk /= COSMO_CLASS["h"]
Plin *= COSMO_CLASS["h"] ** 3
Pnl *= COSMO_CLASS["h"] ** 3

# Fourier transform to get correlation function
rr = np.logspace(-2, 2.6, num=10_000, base=10)
xilin = ct.xi.xi_mm_at_r(rr, kk, Plin, N=2000, step=0.0001)
xi_lin = interp1d(rr, xilin)


def rho_lin_model(x, bias):
    """Infall density profile from linear theory

    Args:
        x (_type_): Radius. Linear scale
    """
    return RHOM * (1 + bias * xi_lin(x))


# Shell mapping model
@njit
def fnl_vol(x, params):
    alpha, beta, xscale, _ = params
    mask = x >= xscale
    xx = (x / xscale - 1) * mask
    return np.power(1 - np.exp(-np.power(xx, beta)), alpha)


@njit
def dfnl_vol(x, params):
    """Analytic derivative"""
    alpha, beta, xscale, _ = params
    mask = x >= xscale
    xx = (x / xscale - 1) * mask
    df = alpha * beta * fnl_vol(x, params)
    df /= xscale * np.power(xx, 1 - beta) * (np.exp(np.power(xx, beta)) - 1)
    return df


def xi_inf_shell(x, params):
    alpha, beta, vsc, bias = params

    # Convert radii to volumes
    v = 4.0 * np.pi * np.power(x, 3) / 3.0

    # Interpolate g_lin over a grid in V_lin
    y = np.logspace(np.log10(vsc), 10, num=1_000, base=10)
    g_lin = interp1d(y * fnl_vol(y, params), y)

    # Evaluate g_lin(V) = V_lin
    v_lin = g_lin(v)

    # Convert `linear` volumes into `linear` radii to evaluate the linear
    # correlation function
    x_lin = np.cbrt(3.0 * v_lin / 4.0 / np.pi)

    # Compute density profile
    rho_inf = rho_lin_model(x_lin, bias)
    # rho_inf = RHOM * (1 + bias* xilin(x_lin))
    rho_inf /= fnl_vol(v_lin, params) + v_lin * dfnl_vol(v_lin, params)

    # Return correlation function
    return rho_inf / RHOM - 1


def xi_inf_post(params, *args):
    """The cost is defined to be:

                    cost = -0.5 * chi^2 + ln|C|

    The covariance C is regulated by an additional parameter `delta`.
    """
    # Unpack data
    x, y, covy, mask, masses, biases = args

    # Unpack parameters
    logA, a, logV, v, logf = params

    mp = 1e14
    alpha = (10**logA * (masses / mp) ** a) * (1 + (masses / mp)) ** -a
    vsc = 10**logV * (masses / mp) ** v
    beta = 1.6 / alpha

    # Check priors
    if np.abs(a) > 1 or np.any(alpha < 0) or np.any(alpha > 8) or np.any(beta < 0) or np.any(vsc < 0) or logf > 3:
        return -np.inf

    # Aggregate likelihood for all mass bins
    lnlike = 0
    for k in range(y.shape[0]):
        p0k = (alpha[k], beta[k], vsc[k], biases[k])

        # Apply mask to data
        xx = x[mask[k, :]]
        yy = y[k, mask[k, :]]
        u = yy - xi_inf_shell(xx, p0k)

        # Add percent error to the covariance - regulated by delta
        cov = covy[k, mask[k, :], :][:, mask[k, :]] + np.diag(np.power(np.exp(logf) * yy, 2))

        # Compute chi squared
        chi2 = np.dot(u, np.linalg.solve(cov, u))

        lnlike -= 0.5 * chi2 + 0.5 * np.log(np.linalg.det(cov))

    # Return cost
    return lnlike
    