import numpy as np
from numba import njit
from scipy.integrate import quad

from src.config import RHOM


@njit()
def rho_orb_model_kernel(x, ainf, rin, rout):
    '''Integrand function in spherical coordinates: model * r^2'''
    # Power-law exponent
    alpha = ainf * (x / rin) / (1 + (x / rin))

    # Model * r^2
    func = np.power(x / rin, -alpha) * np.exp(-0.5 * np.power(x/rout, 2))
    return func * np.power(x, 2)


def rho_orb_model(x, ainf, rin, rout, *args):
    '''Orbiting density profile model. It has the constraint that the integral
    of the density from zero to infinity is equal to the orbiting mass.
    '''
    # Unpack arguments
    (mass, ) = args

    # Constraint on the normalization
    integral = quad(rho_orb_model_kernel, 0, np.inf, args=(ainf, rin, rout))[0]
    A = mass / (4. * np.pi * integral)

    # Power-law exponent
    alpha = ainf * (x / rin) / (1 + (x / rin))

    # Return model
    return A * np.power(x / rin, -alpha) * np.exp(-0.5 * np.power(x / rout, 2))


def rho_orb_log_posterior(params, *data) -> float:
    '''Log-likelihhod function for simultaneously fitting all orbiting density
    profiles. The covariance is regulated by an additional parameter `delta`.
    The model parameters are:

        \alpha_infty = \lambda_\infty * (M / Mp)^(\sigma_\infty)
        r_in         = \lambda_in     * (M / Mp)^(\sigma_r)
        r_out        = \lambda_out    * (M / Mp)^(\sigma_r)

    where Mp is a pivot mass.

    The likelihood is defined to be:

                        L = exp(-0.5 * chi^2) / sqrt(|C|)

    The natural logarithm of the covariance determinant is computed in a `smart`
    way to avoid precision overflow.

    Given the determinant property |bA| = b^2 |A|, where b is a constant,
    and A is a n x n matrix, it follows that if C = bA, the log-determinant:

                        ln|C| = n ln(b) + ln|C/b|

    Args:
        params (_type_): model + likelihood parameters.
        data (*args): data vector and model arguments.
            x (1D):     x data
            y (2D):     y data, each row is a mass bin.
            covy (3D):  covariance matrix, each row is a mass bin.
            mask (2D):  x data mask, each row is a mass bin.
            mass (1D):  mass bin (corrected) mass.
            m_pivot (float): pivot mass

    Returns:
        float: total log-likelihood
    '''
    scale = RHOM**2

    # Unpack data
    x, y, covy, mask, mass, m_pivot = data

    # Check priors
    sa, la, sr, lin, lout, logd = params
    if logd > 0:
        return -np.inf
    delta = np.power(10, logd)

    # Evaluate density profile parameters as power-laws
    rin = np.power(10, sr * np.log10(mass / m_pivot) + np.log10(lin))
    rout = np.power(10, sr * np.log10(mass / m_pivot) + np.log10(lout))
    ainf = np.power(10, sa * np.log10(mass / m_pivot) + np.log10(la))

    # Aggregate likelihood for all mass bins
    lnlike = 0
    for k in range(y.shape[0]):
        # Apply mask to data
        xx = x[mask[k, :]]
        yy = y[k, mask[k, :]]
        u = yy - rho_orb_model(xx, ainf[k], rin[k], rout[k], mass[k])

        # Add percent error to the covariance - regulated by delta
        cov = covy[k, mask[k, :], :][:, mask[k, :]] + \
            np.diag(np.power(delta * yy, 2))

        # Compute chi squared
        chi2 = np.dot(u, np.linalg.solve(cov, u))

        # Compute ln|C| in a 'smart' way
        lndetc = len(u) * np.log(scale) + np.log(np.linalg.det(cov / scale))

        lnlike -= 0.5 * chi2 + 0.5 * lndetc

    return lnlike