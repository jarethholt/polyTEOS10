#!/usr/bin/env python
"""Polynomial approximation to TEOS-10.

This module contains a polynomial approximation to the Thermodynamic Equation
of Seawater 2010 (TEOS-10) library. The functions here calculate density and
related quantities directly from the absolute salinity, conservative
temperature, and pressure (or depth in the Boussinesq case). These direct
calculations were made possible by fitting a 55-term polynomial to the density
within the 'oceanographic funnel' parameter space that describes most seawater.

For details on the polynomial used, see:
    Roquet, F., Madec, G., McDougall, T. J., Barker, P. M., 2014: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling.
For details on TEOS-10 and the oceanographic funnel, see:
    McDougall, T. J., D. R. Jackett, D. G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  Journal of Atmospheric and  Oceanic
    Technology, 20, 730-741.

Original author: Fabien Roquet, Dec 2015
Adapted by: Jareth Holt, May 2019
"""

# Import statements
import numpy as np

# Physical constants
TCELS = 273.15  # 0 Celsius in K
PATM = 101325.  # 1 atm in Pa
GRAV = 9.80665  # Standard gravity, m s-2
CSEA = 3991.86795711963  # Standard heat capacity of seawater, J kg-1 K-1
SAL0 = 35.16504  # Salinity of KCl-normalized seawater, g kg-1
SRED = SAL0 * 40./35.  # Reducing salinity, g kg-1
TRED = 40.  # Reducing temperature, deg C
ZRED = 1e4  # Reducing depth, m
PRED = 1e4  # Reducing pressure, dbar
DBAR2PA = 1e4  # Conversion factor, Pa/dbar
SNDFACTOR = (PRED*DBAR2PA)**.5  # Coefficient in speed of sound, Pa(.5)
RHOBSQ = 1020.  # Reference density for depth-pressure conversion, kg/m3
DTASBSQ = 32.  # Salinity offset for Boussinesq formulation, g kg-1
DTASCMP = 24.  # Salinity offset for compressile formulation, g kg-1
TTP = 273.16  # Triple point temperature in K
PTP = 611.657  # Triple point pressure in Pa


# Number of terms in each polynomial expansion
NSTBSQ = (6, 4, 2, 1)
NSTC75 = (6, 5, 4, 2, 1, 0)
NGWAT = (7, 7, 6, 6, 5, 3, 0)
NGSAL0 = (1,)
NGSAL1 = (1,)
NGSAL2 = (0, 0, 7, 5, 5, 1, 1, 0)

# Global dictionary of coefficients
COEFS = dict()


# Read coefficients file
def readcoefs():
    """Read coefs.txt and save to COEFS.

    Read the file coefs.txt containing the values of the coefficients for each
    of the functions given here. Save the values in the global dictionary
    COEFS.
    """
    global COEFS
    name, data = None, list()
    with open('coefs.txt', mode='rt') as f:
        while True:
            # Check if at end of file
            line = f.readline()
            if not line.strip():
                if (name is not None) and (len(data) > 0):
                    COEFS[name] = tuple(data)
                    break

            if not line.startswith('  '):
                # Header line
                if name is not None:
                    COEFS[name] = tuple(data)
                name = line.strip()
                data = list()
            else:
                # Data line
                data += [float(val) for val in line.split(',') if val.strip()]
    return None


readcoefs()


# Helper functions
def poly1d(z, coefs):
    """Evaluate a 1d polynomial with derivative.

    Evaluate a polynomial in one variable with the given coefficients at the
    given point as well as its first derivative. The point can be given as a
    numpy array, in which case the outputs have the same type and shape as that
    array.

    Arguments:
        z (float or array): Point(s) to evaluate the polynomial at.
        coefs (iterable of float): Coefficients of the polynomial, from lowest
            degree to highest.

    Returns:
        p (float or array): Value of the polynomial at the point(s) `z`.
        dpdz (float or array): Value of the derivative at the point(s) `z`.
    """
    p, dpdz = 0., 0.
    for coef in coefs[-1::-1]:
        dpdz = dpdz*z + p
        p = p*z + coef
    return (p, dpdz)


def poly1d_der(z, dz, coefs):
    """Evaluate a 1d polynomial with derivatives.

    Evaluate a polynomial in ones variable with the given coefficients at the
    given points, or its derivatives to any order.

    The points can be given as numpy arrays as long as they can be broadcast
    against each other; all outputs will have this shape and type.

    Arguments:
        z (float or array): Value(s) of each variable at which to evaluate the
            polynomial.
        dz (int >= 0): Number of derivatives to take with respect to z.
        coefs (iterable of float): Coefficients of the polynomial.

    Returns:
        p (float or array): Value of the polynomial (or derivative) at the
            given point(s).
    """
    p = 0.
    kmax = len(coefs) - 1
    for (k2, coef) in enumerate(coefs[-1::-1]):
        k = kmax - k2
        if k < dz:
            break
        for k1 in range(dz):
            coef *= k-k1
        p = p*z + coef
    return p


def poly2d(y, z, coefs, njs):
    """Evaluate a 2d polynomial with derivatives.

    Evaluate a polynomial in two variables with the given coefficients at the
    given points, as well as the first derivatives with respect to each
    variable.

    The points can be given as numpy arrays as long as they can be broadcast
    against each other; all outputs will have this shape and type.

    The polynomial coefficients are given as a flat iterable (tuple, list, or
    array). They should be ordered from lowest degree to highest, with the
    degree in `y` changing first, followed by the degree for `z`. The input
    `njs` gives the number of degrees of `y` (j) for each degree of `z` (k).

    Arguments:
        y, z (float or array): Value(s) of each variable at which to evaluate
            the polynomial.
        coefs (iterable of float): Coefficients of the polynomial.
        njs (iterable of int): Maximum degree in `y` for each degree of `z`.

    Returns:
        p (float or array): Value of the polynomial at the given point(s).
        dpy, dpz (float or array): Derivative of the polynomial with respect to
            each variable.
    """
    p, dpy, dpz = 0., 0., 0.
    ind = -1
    for jmax in njs[-1::-1]:
        p2, dp2y = 0., 0.
        for j in range(jmax, -1, -1):
            coef = coefs[ind]
            dp2y = dp2y*y + p2
            p2 = p2*y + coef
            ind -= 1
        dpz = dpz*z + p
        p = p*z + p2
        dpy = dpy*z + dp2y
    return (p, dpy, dpz)


def poly2d_der(y, z, dy, dz, coefs, njs):
    """Evaluate a 2d polynomial with derivatives.

    Evaluate a polynomial in two variables with the given coefficients at the
    given points, or its derivatives to any order.

    The points can be given as numpy arrays as long as they can be broadcast
    against each other; all outputs will have this shape and type.

    The polynomial coefficients are given as a flat iterable (tuple, list, or
    array). They should be ordered from lowest degree to highest, with the
    degree in `y` changing first, followed by the degree for `z`. The input
    `njs` gives the number of degrees of `y` (j) for each degree of `z` (k).

    Arguments:
        y, z (float or array): Value(s) of each variable at which to evaluate
            the polynomial.
        dy, dz (int >= 0): Number of derivatives to take with respect to each
            variable.
        coefs (iterable of float): Coefficients of the polynomial.
        njs (iterable of int): Maximum degree in `y` for each degree of `z`.

    Returns:
        p (float or array): Value of the polynomial (or derivative) at the
            given point(s).
    """
    p = 0.
    ind = -1
    kmax = len(njs) - 1
    for (k2, jmax) in enumerate(njs[-1::-1]):
        k = kmax - k2
        if k < dz:
            break
        p2 = 0.
        for j in range(jmax, -1, -1):
            if j < dy:
                ind -= dy
                break
            coef = coefs[ind]
            for j1 in range(dy):
                coef *= (j-j1)
            p2 = p2*y + coef
            ind -= 1
        for k1 in range(dz):
            p2 *= (k-k1)
        p = p*z + p2
    return p


def poly3d(x, y, z, coefs, nijs):
    """Evaluate a 3d polynomial with derivatives.

    Evaluate a polynomial in three variables with the given coefficients at the
    given point, as well as its first derivatives with respect to each
    variable.

    The points can be given as numpy arrays as long as they are broadcastable
    against each other; all outputs will have this shape and type.

    The polynomial coefficients are given as a flat iterable (tuple, list, or
    array). They should be ordered from lowest degree to highest, with the
    degree in `x` changing first, followed by the degrees for `y` and `z`. The
    degrees might therefore be:
        000 100 200 300 010 110 210 ... 002 102 012.
    It is assumed that for each degree `nz` of `z`, there are coefficients for
    every degree of (`x`,`y`) up to the number `nijs[nz]`. In the above
    example, `nijs[0]=3` and `nijs[2]=1`.

    Arguments:
        x, y, z (float or array): Value(s) of the variables at which to
            evaluate the polynomial.
        coefs (iterable of float): Coefficients of the polynomial.
        nijs (iterable of int): Maximum degree of (x,y) for each degree of z.

    Returns:
        p (float or array): Value of the polynomial at the given point(s).
        dpx, dpy, dpz (float or array): Derivative of the polynomial with
            respect to each variable.
    """
    p, dpx, dpy, dpz = 0., 0., 0., 0.
    ind = -1
    for ijmax in nijs[-1::-1]:
        p2, dp2x, dp2y = 0., 0., 0.
        for j in range(ijmax, -1, -1):
            p3, dp3x = 0., 0.
            for i in range(ijmax-j, -1, -1):
                coef = coefs[ind]
                dp3x = dp3x*x + p3
                p3 = p3*x + coef
                ind -= 1
            dp2y = dp2y*y + p2
            p2 = p2*y + p3
            dp2x = dp2x*y + dp3x
        dpz = dpz*z + p
        p = p*z + p2
        dpy = dpy*z + dp2y
        dpx = dpx*z + dp2x
    return (p, dpx, dpy, dpz)


def poly3d_der(x, y, z, dx, dy, dz, coefs, nijs):
    """Evaluate a 3d polynomial with derivatives.

    Evaluate a polynomial in three variables with the given coefficients at the
    given point, or its derivatives to any order.

    The points can be given as numpy arrays as long as they are broadcastable
    against each other; all outputs will have this shape and type.

    The polynomial coefficients are given as a flat iterable (tuple, list, or
    array). They should be ordered from lowest degree to highest, with the
    degree in `x` changing first, followed by the degrees for `y` and `z`. The
    degrees might therefore be:
        000 100 200 300 010 110 210 ... 002 102 012.
    It is assumed that for each degree `nz` of `z`, there are coefficients for
    every degree of (`x`,`y`) up to the number `nijs[nz]`. In the above
    example, `nijs[0]=3` and `nijs[2]=1`.

    Arguments:
        x, y, z (float or array): Value(s) of the variables at which to
            evaluate the polynomial.
        dx, dy, dz (int >= 0): Number of derivatives to take with respect to
            each variable.
        coefs (iterable of float): Coefficients of the polynomial.
        nijs (iterable of int): Maximum degree of (x,y) for each degree of z.

    Returns:
        p (float or array): Value of the polynomial (or derivative) at the
            given point(s).
    """
    p = 0.
    ind = -1
    kmax = len(nijs) - 1
    for (k2, ijmax) in enumerate(nijs[-1::-1]):
        k = kmax - k2
        if k < dz:
            break
        p2 = 0.
        for j in range(ijmax, -1, -1):
            if j < dy:
                # How many entries are left?
                for j1 in range(dy):
                    ind -= ijmax+1 - j1
                break
            p3 = 0.
            for i in range(ijmax-j, -1, -1):
                if i < dx:
                    ind -= dx
                    break
                coef = coefs[ind]
                for i1 in range(dx):
                    coef *= i-i1
                p3 = p3*x + coef
                ind -= 1
            for j1 in range(dy):
                p3 *= j-j1
            p2 = p2*y + p3
        for k1 in range(dz):
            p2 *= k-k1
        p = p*z + p2
    return p


# Boussinesq functions
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
    r0, dr0z = poly1d(zet, COEFS['BSQ0'])

    # Density anomaly
    r1, dr1s, dr1t, dr1z = poly3d(sig, tau, zet, COEFS['BSQ1'], NSTBSQ)

    # Calculate physically-relevant quantities
    rho = r0 + r1
    absq = -dr1t/TRED
    bbsq = dr1s/(2*sig*SRED)
    csnd = (RHOBSQ*GRAV*ZRED/(dr0z + dr1z))**.5
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
    (1027.4514038946588,
     0.17964940584630093,
     0.7655544973485031,
     1500.2088410430842)
    """
    # Reduced variables
    sig = ((salt+DTASBSQ)/SRED)**.5
    tau = tcon / TRED
    zet = dpth / ZRED

    # Vertical reference profile of density
    r1, dr1z = poly1d(zet, COEFS['STIF0'])

    # Normalized density
    rdot, drdots, drdott, drdotz = poly3d(
        sig, tau, zet, COEFS['STIF1'], NSTBSQ)

    # Return all physical quantities
    rho = r1 * rdot
    absq = -r1/TRED * drdott
    bbsq = r1/(2*sig*SRED) * drdots
    csnd = (RHOBSQ*GRAV*ZRED/(rdot*dr1z + r1*drdotz))**.5
    return (rho, absq, bbsq, csnd)


def calnsq_bsq(absq, bbsq, dctdz, dsadz):
    """Calculate the Boussinesq stratification.

    Calculate the square of the buoyancy frequency for a Boussinesq system,
    i.e. the stratification. Here, absq and bbsq are the modified thermal
    expansion and haline contraction coefficients returned by either eos_bsq or
    eos_stif, which are both Boussinesq equations of state.

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
    >>> calnsq_bsq(.18,.77,2e-3,-5e-3)
    4.0476467156862744e-05

    >>> __, absq, bbsq, __ = eos_bsq(30.,10.,1e3)
    >>> calnsq_bsq(absq,bbsq,2e-3,-5e-3)
    4.025600421032772e-05

    >>> __, absq, bbsq, __ = eos_stif(30.,10.,1e3)
    >>> calnsq_bsq(absq,bbsq,2e-3,-5e-3)
    4.025602230274386e-05
    """
    nsq = GRAV/RHOBSQ * (absq*dctdz - bbsq*dsadz)
    return nsq


# Compressible functions
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
    v0, dv0p = poly1d(phi, COEFS['CMP'])

    # Density anomaly
    dta, ddtas, ddtat, ddtap = poly3d(sig, tau, phi, COEFS['C55'], NSTBSQ)

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
    v0, dv0p = poly1d(phi, COEFS['CMP'])

    # Density anomaly
    dta, ddtas, ddtat, ddtap = poly3d(sig, tau, phi, COEFS['C75'], NSTC75)

    # Return all physical quantities
    svol = v0 + dta
    alpha = ddtat/TRED / svol
    beta = -ddtas/(2*sig*SRED) / svol
    csnd = svol * (-(dv0p + ddtap))**(-.5) * SNDFACTOR
    return (svol, alpha, beta, csnd)


# Gibbs function formulation
def gibbs_wat(temp, pres, dt, dp, orig=False):
    """Calculate the Gibbs function for pure water.

    Calculate the Gibbs function for pure water from the in-situ temperature
    and pressure, or its derivatives with respect to these variables.

    Arguments:
        temp (float or array): In-situ temperature in degrees Celsius.
        pres (float or array): Seawater pressure in dbar (absolute pressure
            minus 1 atm).
        dt, dp (int >= 0): Number of temperature and pressure derivatives to
            take.
        orig (bool, optional): If True, the original coefficients for the Gibbs
            function are used. If False (default), then adjusted coefficients
            are used which satisfy that the entropy and internal energy of pure
            water are 0 at the triple point.

    Returns:
        g (float or array): Gibbs free energy of pure water, in units of
            (J kg-1) / (deg C)^dt / (dbar)^dp.
    """
    # Calculate reduced variables
    y = temp / TRED
    z = pres / PRED
    # Calculate polynomial
    if dp == 0:
        name0 = 'GWAT0_' + ('ORIG' if orig else 'TRUE')
        g0 = poly1d_der(y, dt, COEFS[name0])
    else:
        g0 = 0.
    g1 = poly2d_der(y, z, dt, dp, COEFS['GWAT1'], NGWAT)
    # Scale value to match derivatives
    g = (g0 + g1) / (TRED**dt * PRED**dp)
    return g


def gibbs_sal(salt, temp, pres, ds, dt, dp, orig=False):
    """Calculate the Gibbs function for salt in seawater.

    Calculate the Gibbs function for salt in seawater from the absolute
    salinity, in-situ temperature, and pressure, or its derivatives with
    respect to these variables.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        temp (float or array): In-situ temperature in degrees Celsius.
        pres (float or array): Seawater pressure in dbar (absolute pressure
            minus 1 atm).
        ds, dt, dp (int >= 0): Number of salinity, temperature, and pressure
            derivatives to take.
        orig (bool, optional): If True, the original coefficients for the Gibbs
            function are used. If False (default), then adjusted coefficients
            are used which satisfy that the entropy and enthalpy of seawater
            are 0 under standard conditions.

    Returns:
        g (float or array): Gibbs free energy of salt, in units of
            (J kg-1) / (g kg-1)^ds / (deg C)^dt / (dbar)^dp.
    """
    # Special case: zero salinity
    """
    if salt == 0.:
        # Can only have specific derivatives
        if any(isinstance(x, np.ndarray) for x in (salt, temp, pres)):
            out = np.ones(np.broadcast(salt, temp, pres).shape)
        else:
            out = 1.
        if ds == 0:
            g = 0. * out
            return g
        elif ds > 1:
            g = np.nan * out
            return g
        else:
            if (dt > 1) or (dp > 0):
                # Calculate g2_tp
                y = temp / TRED
                z = pres / PRED
                jkmax = NGSAL2[2]
                njs = tuple((jkmax+1-k) for k in range(jkmax+1))
                njk = ((jkmax+1)*(jkmax+2))//2
                g2 = poly2d_der(y, z, dt, dp, COEFS['GSAL2'][2:njk+2], njs)
                g = g2 / SRED / TRED**dt / PRED**dp
                return g
            else:
                g = np.nan*out
                return g
    """

    # Calculate reduced variables
    x = (salt / SRED)**.5
    y = temp / TRED
    z = pres / PRED
    # Calculate logarithmic salinity term
    g0yz = poly2d_der(y, z, dt, dp, COEFS['GSAL0'], (1,))/2/SRED
    if ds == 0:
        g0 = g0yz * salt * np.log(salt/SRED)
    elif ds == 1:
        g0 = g0yz * (1 + np.log(salt/SRED))
    else:
        sinv = 1./salt
        g0 = g0yz * sinv
        for i in range(1, ds-1):
            g0 *= -i*sinv
    # Calculate the linear, adjustable salinity terms
    if (ds > 1) or (dt > 1) or (dp > 0):
        g1 = 0.
    else:
        name1 = 'GSAL1_' + ('ORIG' if orig else 'TRUE')
        g1 = poly1d_der(y, dt, COEFS[name1]) * salt**(ds == 0) / SRED
    # Calculate the other salinity terms
    g2 = 0.
    imax = len(NGSAL2) - 1
    ind = 0
    for (i1, jkmax) in enumerate(NGSAL2[-1::-1]):
        i = imax-i1
        njs = tuple((jkmax-k) for k in range(jkmax+1))
        njk = ((jkmax+1)*(jkmax+2))//2
        inds = slice(ind-njk, (ind if ind < 0 else None))
        coefs = COEFS['GSAL2'][inds]
        g2jk = poly2d_der(y, z, dt, dp, coefs, njs)
        for i1 in range(ds):
            g2jk *= .5*(i-2*i1)/salt
        g2 = g2*x + g2jk
        ind -= njk
    # Scale value to match derivatives
    g = (g0 + g1 + g2) / TRED**dt / PRED**dp
    return g


def caladjustedcoefs(write=True):
    """Calculate the adjusted coefficients for the Gibbs functions.

    Calculate the adjusted coefficients for the Gibbs functions of pure water
    and salt. Using the new coefficients, the following conditions are
    satisfied to within the current numerical precision:
        - The entropy and internal energy of pure water are 0 at the triple
          point temperature (273.16 K) and pressure (611.657 Pa); and
        - The entropy and enthalpy of seawater are 0 under standard salinity
          (35.16504 g kg-1), temperature (273.15 K) and pressure (101325 Pa).

    Arguments:
        write (bool, optional): If True (default) and the adjusted coefficients
            are not listed in `coefs.txt`, then the values calculated here will
            be appended. If False or if the entries already exist, nothing will
            be done.

    Returns:
        gw00_true, gw10_true, gs100_true, gs110_true (float): The adjusted
            values for these coefficients in the Gibbs polynomial.
    """
    # Gibbs function for pure water
    gw00_orig, gw10_orig = COEFS['GWAT0_ORIG'][:2]
    ttp = TTP - TCELS
    ptp = (PTP - PATM)/DBAR2PA
    gwt_orig = gibbs_wat(ttp, ptp, 0, 0, orig=True)
    etawt_orig = -gibbs_wat(ttp, ptp, 1, 0, orig=True)
    volwt_orig = gibbs_wat(ttp, ptp, 0, 1, orig=True)
    uwt_orig = gwt_orig + TTP*etawt_orig - PTP/DBAR2PA*volwt_orig
    gw10_true = gw10_orig + TRED*etawt_orig
    gw00_true = gw00_orig + TCELS*etawt_orig + uwt_orig

    # Gibbs function for salt
    gs100_orig, gs110_orig = COEFS['GSAL1_ORIG'][:2]
    gs0_curr = gibbs_sal(SAL0, 0., 0., 0, 0, 0, orig=True)
    etas0_curr = -gibbs_sal(SAL0, 0., 0., 0, 1, 0, orig=True)
    gs110_true = gs110_orig + SRED/SAL0*(TRED*etas0_curr - gw10_true)
    gs100_true = gs100_orig - SRED/SAL0*(gw00_true + gs0_curr)

    # Write outputs
    if write:
        fmt = '{:+21.14e}'
        names = ('GWAT0_TRUE', 'GSAL1_TRUE')
        coefs = ((gw00_true, gw10_true), (gs100_true, gs110_true))
        for (name, cs) in zip(names, coefs):
            if name not in COEFS:
                with open('coefs.txt', mode='at') as f:
                    f.write(f'{name}\n  ')
                    f.write(', '.join(fmt.format(c) for c in cs))
                    f.write('\n')
    return (gw00_true, gw10_true, gs100_true, gs110_true)


# Main script: Run doctest
if __name__ == '__main__':
    import doctest
    doctest.testmod()
