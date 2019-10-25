#!/usr/bin/env python3
"""Boussinesq equations of state.

This module provides two functions for the equation of state (EOS) of seawater
suitable for Boussinesq ocean models. In both cases, the thermodynamic
variables are absolute salinity, conservative temperature, and depth. In
comparison, the standard formulation of an EOS is in terms of absolute
salinity, in-situ temperature, and pressure.

These equations of state are given as polynomials in temperature, pressure, and
a salinity-related variable. The evaluation of polynomials and their
derivatives is implemented in the `poly` module.
"""

# Import statements
import numpy as np
import aux
from const import GRAV, SRED, TRED, ZRED, RHOBSQ, DTASBSQ, NPZFILE

# Load the relevant coefficients
with np.load(NPZFILE) as cdata:
    CBSQ0, CBSQ1, IJBSQ, CSTIF0, CSTIF1, IJSTIF = [
        cdata[name] for name in ('BSQ0', 'BSQ1', 'ijmaxs_BSQ1', 'STIF0',
                                 'STIF1', 'ijmaxs_STIF1')]


# Functions
def eos_bsq(salt, tcon, dpth):
    """Calculate Boussinesq density.

    Calculate the density and related quantities using a Boussinesq
    approximation from the salinity, temperature, and pressure. In the
    Boussinesq form, any term divided by the full density should use the
    reference density RHOBSQ (1020 kg m-3).

    The points can be given as numpy arrays as long as they are broadcastable
    against each other; all outputs will have this shape and type.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        tcon (float or array): Conservative temperature in degrees Celsius.
        dpth (float or array): Seawater depth in m; equivalently, seawater
            reference hydrostatic pressure in dbar.

    Returns:
        rho (float or array): Density in kg m-3.
        absq (float or array): Modified thermal expansion coefficient
            (-drho/dtcon) in kg m-3 K-1.
        bbsq (float or array): Modified haline contraction coefficient
            (drho/dsalt) in kg m-3 (g kg-1)-1.
        csnd (float or array): Speed of sound in m s-1.

    Examples
    --------
    >>> eos_bsq(30.,10.,1e3)  #doctest: +NORMALIZE_WHITESPACE
    (1027.4514011715235,
     0.17964628133829566,
     0.7655553707894517,
     1500.2086843982124)
    """
    # Reduced variables
    sig = ((salt+DTASBSQ)/SRED)**.5
    tau = tcon / TRED
    zet = dpth / ZRED

    # Vertical reference profile of density
    r0, r0z = aux.poly1d_1der(zet, CBSQ0)

    # Density anomaly
    r1, r1s, r1t, r1z = aux.poly3d_1der(sig, tau, zet, CBSQ1, IJBSQ)

    # Calculate physically-relevant quantities
    rho = r0 + r1
    absq = -r1t/TRED
    bbsq = r1s/(2*sig*SRED)
    csnd = (RHOBSQ*GRAV*ZRED/(r0z + r1z))**.5
    return (rho, absq, bbsq, csnd)


def eos_stif(salt, tcon, dpth):
    """Calculate stiffened density.

    Calculate the density and related quantities using a stiffened Boussinesq
    approximation from the salinity, temperature, and pressure. In the
    Boussinesq form, any term divided by the full density should use the
    reference density RHOBSQ (1020 kg m-3).

    The points can be given as numpy arrays as long as they are broadcastable
    against each other; all outputs will have this shape and type.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        tcon (float or array): Conservative temperature in degrees Celsius.
        dpth (float or array): Seawater depth in m; equivalently, seawater
            reference hydrostatic pressure in dbar.

    Returns:
        rho (float or array): Density in kg m-3.
        absq (float or array): Modified thermal expansion coefficient
            (-drho/dtcon) in kg m-3 K-1.
        bbsq (float or array): Modified haline contraction coefficient
            (drho/dsalt) in kg m-3 (g kg-1)-1.
        csnd (float or array): Speed of sound in m s-1.

    Examples
    --------
    >>> eos_stif(30.,10.,1e3)  #doctest: +NORMALIZE_WHITESPACE
    (1027.4514038962773,
     0.1796494059656094,
     0.7655544988472869,
     1500.2088411949183)
    """
    # Reduced variables
    sig = ((salt+DTASBSQ)/SRED)**.5
    tau = tcon / TRED
    zet = dpth / ZRED

    # Vertical reference profile of density
    r1, r1z = aux.poly1d_1der(zet, CSTIF0)

    # Normalized density
    rdot, rdots, rdott, rdotz = aux.poly3d_1der(sig, tau, zet, CSTIF1, IJSTIF)

    # Return all physical quantities
    rho = r1 * rdot
    absq = -r1/TRED * rdott
    bbsq = r1/(2*sig*SRED) * rdots
    csnd = (RHOBSQ*GRAV*ZRED/(rdot*r1z + r1*rdotz))**.5
    return (rho, absq, bbsq, csnd)


def calnsq(absq, bbsq, dctdz, dsadz):
    """Calculate the Boussinesq stratification.

    Calculate the square of the buoyancy frequency for a Boussinesq system,
    i.e. the stratification. Here, absq and bbsq are the modified thermal
    expansion and haline contraction coefficients returned by either `eos_bsq`
    or `eos_stif`, which are both Boussinesq equations of state.

    The points can be given as numpy arrays as long as they are broadcastable
    against each other; all outputs will have this shape and type. In
    particular, the expansion and contraction coefficients have to be on the
    same vertical grid as the temperature and salinity gradients.

    Arguments:
        absq (float or array): Modified thermal expansion coefficient
            (-drho/dtcon) in kg m-3 K-1.
        bbsq (float or array): Modified haline contraction coefficient
            (drho/dsalt) in kg m-3 (g kg-1)-1.
        dctdz (float or array): Vertical gradient of the conservative
            temperature in K m-1.
        dsadz (float or array): Vertical gradient of the absolute salinity
            in g kg-1 m-1.

    Returns:
        nsq (float or array): Squared buoyancy frequency in s-2.

    Examples
    --------
    >>> calnsq(.18,.77,2e-3,-5e-3)
    4.0476467156862744e-05

    >>> __, absq, bbsq, __ = eos_bsq(30.,10.,1e3)
    >>> calnsq(absq,bbsq,2e-3,-5e-3)
    4.025600421032772e-05

    >>> __, absq, bbsq, __ = eos_stif(30.,10.,1e3)
    >>> calnsq(absq,bbsq,2e-3,-5e-3)
    4.0256022377087266e-05
    """
    nsq = GRAV/RHOBSQ * (absq*dctdz - bbsq*dsadz)
    return nsq


# Main script: Run doctest
if __name__ == '__main__':
    import doctest
    doctest.testmod()
