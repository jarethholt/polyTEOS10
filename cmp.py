#!/usr/bin/env python3
"""Compressible equations of state.

This module provides two functions for the equation of state (EOS) of seawater
suitable for compressible ocean models. In both cases, the thermodynamic
variables are absolute salinity, conservative temperature, and pressure. In
comparison, the standard formulation of an EOS is in terms of absolute
salinity, in-situ temperature, and pressure.

These equations of state are given as polynomials in temperature, pressure, and
a salinity-related variable. The evaluation of polynomials and their
derivatives is implemented in the `poly` module.
"""

# Import statements
import numpy as np
import aux
from const import SRED, TRED, PRED, SNDFACTOR, DTASCMP, NPZFILE

# Load the relevant coefficients
with np.load(NPZFILE) as cdata:
    CMP, C55, IJ55, C75, IJ75 = [
        cdata[name] for name in ('CMP', 'C55', 'ijmaxs_C55', 'C75',
                                 'ijmaxs_C75')]


# Main functions
def eos_c55(salt, tcon, pres):
    """Calculate specific volume (55-term polynomial).

    Calculate the specific volume and related quantities using a 55-term
    polynomial fit. This equation of state is compressible and thus uses
    pressure rather than depth.

    The points can be given as numpy arrays as long as they are broadcastable
    against each other; all outputs will have this shape and type.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        tcon (float or array): Conservative temperature in degrees Celsius.
        pres (float or array): Seawater pressure in dbar (absolute pressure
            minus 1 atm).

    Returns:
        svol (float or array): Specific volume (1/density) in m3 kg-1.
        alpha (float or array): Thermal expansion coefficient K-1.
        beta (float or array): Haline contraction coefficient in (g kg-1)-1.
        csnd (float or array): Speed of sound in m s-1.

    Examples
    --------
    >>> eos_c55(30.,10.,1e3)  #doctest: +NORMALIZE_WHITESPACE
    (0.0009732820466146228,
     0.00017485531216005049,
     0.0007450974030844275,
     1500.0006086791234)
    """
    # Reduced variables
    sig = ((salt+DTASCMP)/SRED)**.5
    tau = tcon / TRED
    phi = pres / PRED

    # Vertical reference profile of density
    v0, dv0p = aux.poly1d_1der(phi, CMP)

    # Density anomaly
    dta, ddtas, ddtat, ddtap = aux.poly3d_1der(sig, tau, phi, C55, IJ55)

    # Return all physical quantities
    svol = v0 + dta
    alpha = ddtat/TRED / svol
    beta = -ddtas/(2*sig*SRED) / svol
    csnd = svol * (-(dv0p + ddtap))**(-.5) * SNDFACTOR
    return (svol, alpha, beta, csnd)


def eos_c75(salt, tcon, pres):
    """Calculate specific volume (75-term polynomial).

    Calculate the specific volume and related quantities using a 75-term
    polynomial fit. This equation of state is compressible and thus uses
    pressure rather than depth.

    The points can be given as numpy arrays as long as they are broadcastable
    against each other; all outputs will have this shape and type.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        tcon (float or array): Conservative temperature in degrees Celsius.
        pres (float or array): Seawater pressure in dbar (absolute pressure
            minus 1 atm).

    Returns:
        svol (float or array): Specific volume (1/density) in m3 kg-1.
        alpha (float or array): Thermal expansion coefficient K-1.
        beta (float or array): Haline contraction coefficient in (g kg-1)-1.
        csnd (float or array): Speed of sound in m s-1.

    Examples
    --------
    >>> eos_c75(30.,10.,1e3)  #doctest: +NORMALIZE_WHITESPACE
    (0.0009732819627722665,
     0.00017484355352401316,
     0.000745119667788285,
     1500.0067343599003)
    """
    # Reduced variables
    sig = ((salt+DTASCMP)/SRED)**.5
    tau = tcon / TRED
    phi = pres / PRED

    # Vertical reference profile of density
    v0, dv0p = aux.poly1d_1der(phi, CMP)

    # Density anomaly
    dta, ddtas, ddtat, ddtap = aux.poly3d_1der(sig, tau, phi, C75, IJ75)

    # Return all physical quantities
    svol = v0 + dta
    alpha = ddtat/TRED / svol
    beta = -ddtas/(2*sig*SRED) / svol
    csnd = svol * (-(dv0p + ddtap))**(-.5) * SNDFACTOR
    return (svol, alpha, beta, csnd)


# Main script: Run doctest
if __name__ == '__main__':
    import doctest
    doctest.testmod()
