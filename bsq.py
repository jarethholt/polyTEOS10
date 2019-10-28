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


# Equation of state functions
def eos_bsq0(dpth):
    """Calculate Boussinesq density reference profile.

    Calculate the reference profile for Boussinesq density, i.e. the principal
    component of density that depends only on depth.

    Arguments:
        dpth (float or array): Seawater depth in m; equivalently, seawater
            reference hydrostatic pressure in dbar.

    Returns:
        r0 (float or array): Density in kg m-3.
        r0z (float or array): Derivative of density with respect to depth, in
            units of kg m-3 m-1.
    """
    zet = dpth / ZRED
    r0, r0z = aux.poly1d_1der(zet, CBSQ0)
    return (r0, r0z)


def eos_bsq1(salt, tcon, dpth):
    """Calculate Boussinesq density anomaly.

    Calculate the anomaly from the reference profile for Boussinesq density.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        tcon (float or array): Conservative temperature in degrees Celsius.
        dpth (float or array): Seawater depth in m; equivalently, seawater
            reference hydrostatic pressure in dbar.

    Returns:
        r1 (float or array): Density anomaly in kg m-3.
        r1s, r1t, r1z (float or array): Derivatives of the density anomaly with
            respect to salinity, temperature, and depth.
    """
    sig = ((salt+DTASBSQ)/SRED)**.5
    tau = tcon / TRED
    zet = dpth / ZRED
    r1, r1s, r1t, r1z = aux.poly3d_1der(sig, tau, zet, CBSQ1, IJBSQ)
    return (r1, r1s, r1t, r1z)


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
    # Calculate reference profile and anomaly of density
    r0, r0z = eos_bsq0(dpth)
    r1, r1s, r1t, r1z = eos_bsq1(salt, tcon, dpth)

    # Calculate physically-relevant quantities
    rho = r0 + r1
    absq = -r1t/TRED
    bbsq = r1s / (2 * ((salt+DTASBSQ)*SRED)**.5)
    csnd = (RHOBSQ*GRAV*ZRED/(r0z + r1z))**.5
    return (rho, absq, bbsq, csnd)


def eos_stif0(dpth):
    """Calculate stiffened density reference profile.

    Calculate the reference profile of stiffened Boussinesq density, i.e. the
    principal depth-dependent component.

    Arguments:
        dpth (float or array): Seawater depth in m; equivalently, seawater
            reference hydrostatic pressure in dbar.

    Returns:
        r1 (float or array): Reference profile of density in kg m-3.
        r1z (float or array): Derivative of the reference profile with respect
            to depth, in units of kg m-3 m-1.
    """
    zet = dpth / ZRED
    r1, r1z = aux.poly1d_1der(zet, CSTIF0)
    return (r1, r1z)


def eos_stif1(salt, tcon, dpth):
    """Calculate stiffened density scaling factor.

    Calculate the scaling factor of the stiffened density, the multiplicative
    correction to the reference profile due to salinity and temperature.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        tcon (float or array): Conservative temperature in degrees Celsius.
        dpth (float or array): Seawater depth in m; equivalently, seawater
            reference hydrostatic pressure in dbar.

    Returns:
        rdot (float or array): Density scaling factor, unitless.
        rdots, rdott, rdotz (float or array): Derivatives of the scaling factor
            with respect to salinity, temperature, and depth.
    """
    sig = ((salt+DTASBSQ)/SRED)**.5
    tau = tcon / TRED
    zet = dpth / ZRED
    rdot, rdots, rdott, rdotz = aux.poly3d_1der(sig, tau, zet, CSTIF1, IJSTIF)
    return (rdot, rdots, rdott, rdotz)


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
    # Calculate reference profile and scaling factor
    r1, r1z = eos_stif0(dpth)
    rdot, rdots, rdott, rdotz = eos_stif1(salt, tcon, dpth)

    # Return all physical quantities
    rho = r1 * rdot
    absq = -r1/TRED * rdott
    bbsq = r1*rdots / (2 * ((salt+DTASBSQ)*SRED)**.5)
    csnd = (RHOBSQ*GRAV*ZRED/(rdot*r1z + r1*rdotz))**.5
    return (rho, absq, bbsq, csnd)


# Additional functions
def stratification(absq, bbsq, dctdz, dsadz):
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
    >>> stratification(.18,.77,2e-3,-5e-3)
    4.0476467156862744e-05

    >>> __, absq, bbsq, __ = eos_bsq(30.,10.,1e3)
    >>> stratification(absq,bbsq,2e-3,-5e-3)
    4.025600421032772e-05

    >>> __, absq, bbsq, __ = eos_stif(30.,10.,1e3)
    >>> stratification(absq,bbsq,2e-3,-5e-3)
    4.0256022377087266e-05
    """
    nsq = GRAV/RHOBSQ * (absq*dctdz - bbsq*dsadz)
    return nsq


def potenergy_bsq0(dpth):
    """Calculate the Boussinesq potential energy reference profile.

    Calculate the potential energy in the Boussinesq case corresponding to the
    reference profile of density.

    Arguments:
        dpth (float or array): Seawater depth in m; equivalently, seawater
            reference hydrostatic pressure in dbar.

    Returns:
        epot0 (float or array): Potential energy in J m-3.
    """
    # Construct coefficients of depth integral
    kmax = CBSQ0.size - 1
    cep0 = np.zeros(kmax+2)
    cep0[1:] = CBSQ0 / np.arange(1, kmax+2)

    # Evaluate polynomial
    zet = dpth / ZRED
    epot0 = aux.poly1d(zet, cep0) * ZRED
    return epot0


def potenergy_bsq1(salt, tcon, dpth):
    """Calculate the Boussinesq potential energy anomaly.

    Calculate the potential energy in the Boussinesq case corresponding to the
    density anomaly.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        tcon (float or array): Conservative temperature in degrees Celsius.
        dpth (float or array): Seawater depth in m; equivalently, seawater
            reference hydrostatic pressure in dbar.

    Returns:
        epot1 (float or array): Potential energy in J m-3.
    """
    # Construct coefficients of the depth integral
    kmax = IJBSQ.size - 1
    ijep1 = np.zeros(kmax+2, dtype=int)
    ijep1[1:] = IJBSQ
    cep1 = np.zeros(CBSQ1.size+1)
    cep1[1:] = CBSQ1
    ind = 1
    for k in range(1, kmax+2):
        ijmax = ijep1[k]
        nij = (ijmax+1)*(ijmax+2)//2
        cep1[ind:ind+nij] /= k
        ind += nij

    # Evaluate polynomial
    sig = ((salt+DTASBSQ)/SRED)**.5
    tau = tcon / TRED
    zet = dpth / ZRED
    epot1 = aux.poly3d(sig, tau, zet, cep1, ijep1) * ZRED
    return epot1


# Main script: Run doctest
if __name__ == '__main__':
    import doctest
    doctest.testmod()
