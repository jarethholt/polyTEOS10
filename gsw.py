#!/usr/bin/env python
"""Gibbs seawater toolbox.

This module contains a polynomial approximation to the Thermodynamic Equation
of Seawater 2010 (TEOS-10) library. The functions here calculate the Gibbs free
energy of seawater from the absolute salinity, in-situ temperature, and
seawater (gauge) pressure. From the Gibbs energy, other quantities such as
density or potential temperature can be derived.
"""

# Import statements
import numpy as np
import aux
from const import (TCELS, PATM, TTP, PTP, DBAR2PA, CSEA, SAL0, SRED, TRED,
                   PRED, SMIN, TMIN, ENTMIN, NPZFILE)

# Load the relevant coefficients
with np.load(NPZFILE) as cdata:
    CWAT0_ORIG, CWAT0_ADJ, CWAT1 = [
        cdata[f'GWAT{suf}'] for suf in ('0_ORIG', '0_ADJ', '1')]
    CSAL0, CSAL1_ORIG, CSAL1_ADJ, CSAL2 = [
        cdata[f'GSAL{suf}'] for suf in ('0', '1_ORIG', '1_ADJ', '2')]
    JWAT, JKSAL = cdata['jmaxs_GWAT1'], cdata['jkmaxs_GSAL2']


# Primary Gibbs functions
def gibbs_wat(temp, pres, dmax, orig=False):
    """Calculate the Gibbs function for pure water.

    Calculate the Gibbs function for pure water and its derivatives up to a
    given order from the in-situ temperature and gauge (seawater) pressure.

    Arguments:
        temp (float or array): In-situ temperature in degrees Celsius.
        pres (float or array): Gauge (seawater) pressure in dbar
            (absolute pressure minus 1 atm).
        dmax (int >= 0): Maximum number of temperature and pressure derivatives
            to calculate.
        orig (bool, optional): If True, the original coefficients for the Gibbs
            function are used. If False (default), then adjusted coefficients
            are used which satisfy that the entropy and internal energy of pure
            water are 0 at the triple point.

    Returns:
        gs (list of list of float or array): Gibbs free energy of pure water
            and its derivatives, in units of
                (J kg-1) / (deg C)^dt / (dbar)^dp.
            The derivatives are returned with the temperature derivatives
            increasing first; for example, with dtpmax=3:
                gs = [[g, gt, gtt, gttt], [gp, gtp, gttp],
                      [gpp, gtpp], [gppp]]
    """
    # Calculate reduced variables
    y = temp / TRED
    z = pres / PRED

    # Calculate higher-order polynomial
    gs = aux.poly2d_ders(y, z, CWAT1, JWAT, dmax, yscale=TRED, zscale=PRED)

    # Add leading polynomial
    coefs0 = (CWAT0_ORIG if orig else CWAT0_ADJ)
    g0, g0t = aux.poly1d_1der(y, coefs0, zscale=TRED)
    gs[0][0] += g0
    gs[0][1] += g0t
    return gs


def gibbs_wat0(temp, dmax, orig=False):
    """Calculate the Gibbs function for pure water at 1 atm.

    Calculate the Gibbs function for pure water from the in-situ temperature
    at a pressure of 1 atm and its derivatives with respect to both temperature
    and pressure.

    Arguments:
        temp (float or array): In-situ temperature in degrees Celsius.
        dmax (int >= 0): Maximum number of temperature and pressure derivatives
            to calculate.
        orig (bool, optional): If True, the original coefficients for the Gibbs
            function are used. If False (default), then adjusted coefficients
            are used which satisfy that the entropy and internal energy of pure
            water are 0 at the triple point.

    Returns:
        gs (list of list of float or array): Gibbs free energy of pure water
            and its derivatives, in units of
                (J kg-1) / (deg C)^dt / (dbar)^dp.
            The derivatives are returned with the temperature derivatives
            increasing first; for example, with dtpmax=3:
                gs = [[g, gt, gtt, gttt], [gp, gtp, gttp],
                      [gpp, gtpp], [gppp]]
    """
    # Calculate reduced variable, initialize arrays
    y = temp / TRED
    gs = list()

    # Calculate higher-order polynomials
    kmax = len(JWAT) - 1
    ind0 = 0
    for dp in range(min(dmax, kmax) + 1):
        # Construct the coefficients for this pressure derivative
        ind1 = JWAT[dp] + 1
        coefs = np.array(CWAT1[ind0:ind0+ind1])
        for ip in range(1, dp+1):
            coefs *= ip/PRED
        ind0 += ind1

        # Calculate the polynomial in temperature
        g1ys = aux.poly1d_ders(y, coefs, dmax-dp, zscale=TRED)
        # Add the results to gs
        gs.append(g1ys)

    # Add the leading-order polynomial
    coefs = (CWAT0_ORIG if orig else CWAT0_ADJ)
    g0, g0t = aux.poly1d_1der(y, coefs, zscale=TRED)
    gs[0][0] += g0
    if dmax > 0:
        gs[0][1] += g0t
    return gs


def gibbs_sal(salt, temp, pres, dtpmax, orig=False):
    """Calculate the Gibbs function for salt in seawater.

    Calculate the Gibbs function for salt in seawater from the absolute
    salinity, in-situ temperature, and seawater (gauge) pressure and its
    derivatives with respect to temperature and pressure.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        temp (float or array): In-situ temperature in degrees Celsius.
        pres (float or array): Seawater pressure in dbar (absolute pressure
            minus 1 atm).
        dtpmax (int >= 0): Maximum number of (t,p)-derivatives to take.
        orig (bool, optional): If True, the original coefficients for the Gibbs
            function are used. If False (default), then adjusted coefficients
            are used which satisfy that the entropy and enthalpy of seawater
            are 0 under standard conditions.

    Returns:
        gs (list of list of list of float or array): Gibbs free energy of salt
            in units of
                (J kg-1) / (g kg-1)^ds / (deg C)^dt / (dbar)^dp.
            The derivatives are returned with those in temperature varying
            first followed by pressure. For dtpmax=3, this gives:
                gs = [[g, gt, gtt, gttt], [gp, gtp, gttp], [gpp, gtpp],
                      [gppp]].
    """
    # Calculate reduced variables
    s1 = salt/SRED
    x = s1**.5
    s0 = np.maximum(s1, SMIN)
    smask = (s1 >= SMIN)
    y = temp / TRED
    z = pres / PRED

    # Set up the list of values
    scalar = aux.isscalar(x, y, z)
    out = np.broadcast(x, y, z)
    gs = [[np.zeros(out.shape) for dt in range(dtpmax+1-dp)]
          for dp in range(dtpmax+1)]

    # Calculate high-order polynomial terms
    ind = 0
    for jkmax in JKSAL[::-1]:
        # Find the coefficients for this term
        njk = (jkmax+1)*(jkmax+2)//2
        jmaxs = np.arange(jkmax, -1, -1)
        coefs = CSAL2[ind-njk:(ind if (ind != 0) else None)]
        ind -= njk

        # Calculate this polynomial in y and z
        gyzs = aux.poly2d_ders(
            y, z, coefs, jmaxs, dtpmax, yscale=TRED, zscale=PRED)

        # Add this contribution to the totals
        for dp in range(dtpmax+1):
            for dt in range(dtpmax+1-dp):
                gs[dp][dt] *= x
                gs[dp][dt] += gyzs[dp][dt]

    # Add the linear and logarithmic terms
    csal1 = (CSAL1_ORIG if orig else CSAL1_ADJ)
    lnx = np.log(s0)/2*smask
    gs[0][0] += CSAL0[0]*lnx + csal1[0]
    if dtpmax > 0:
        gs[0][1] += (CSAL0[1]*lnx + csal1[1])/TRED

    # Scale everything by one more factor of salinity
    for dp in range(dtpmax+1):
        for dt in range(dtpmax+1-dp):
            gs[dp][dt] *= s1
    # Reformat elements to be floats
    if scalar:
        aux.makescalar(gs)
    return gs


def gibbs_sal0(salt, temp, dtpmax, orig=False):
    """Calculate the Gibbs function for salt in seawater at 1 atm.

    Calculate the Gibbs function for salt in seawater at 1 atm from the
    absolute salinity and in-situ temperature and its derivatives with respect
    to temperature and pressure.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        temp (float or array): In-situ temperature in degrees Celsius.
        dtpmax (int >= 0): Maximum number of (t,p)-derivatives to take.
        orig (bool, optional): If True, the original coefficients for the Gibbs
            function are used. If False (default), then adjusted coefficients
            are used which satisfy that the entropy and enthalpy of seawater
            are 0 under standard conditions.

    Returns:
        gs (list of list of list of float or array): Gibbs free energy of salt
            in units of
                (J kg-1) / (g kg-1)^ds / (deg C)^dt / (dbar)^dp.
            The derivatives are returned with those in temperature varying
            first followed by pressure. For dtpmax=3, this gives:
                gs = [[g, gt, gtt, gttt], [gp, gtp, gttp], [gpp, gtpp],
                      [gppp]].
    """
    # Calculate reduced variables
    s1 = salt/SRED
    x = s1**.5
    s0 = np.maximum(s1, SMIN)
    smask = (s1 >= SMIN)
    y = temp / TRED

    # Set up the list of values
    scalar = aux.isscalar(x, y)
    out = np.broadcast(x, y)
    gs = [[np.zeros(out.shape) for dt in range(dtpmax+1-dp)]
          for dp in range(dtpmax+1)]

    # Calculate high-order polynomial terms
    ind0 = 0
    for jkmax in JKSAL[::-1]:
        njk = (jkmax+1)*(jkmax+2)//2
        ind0 -= njk
        ind1 = 0
        gscurr = list()
        for dp in range(dtpmax+1):
            # Are there any values at this order?
            ind2 = jkmax+1-dp
            if ind2 <= 0:
                # Append zeros
                gscurr.append([0. for dt in range(dtpmax+1-dp)])
                continue

            # Find the coefficients for this pressure derivative
            start = ind0+ind1
            end = start+ind2
            end = (None if (end == 0) else end)
            coefs = np.array(CSAL2[start:end])
            for ip in range(1, dp+1):
                coefs *= ip/PRED
            ind1 += ind2

            # Calculate the polynomial in temperature
            gys = aux.poly1d_ders(y, coefs, dtpmax-dp, zscale=TRED)
            # Add the results to gs
            gscurr.append(gys)

        # Add these contributions to the totals
        for dp in range(dtpmax+1):
            for dt in range(dtpmax+1-dp):
                gs[dp][dt] *= x
                gs[dp][dt] += gscurr[dp][dt]

    # Add the linear and logarithmic terms
    csal1 = (CSAL1_ORIG if orig else CSAL1_ADJ)
    x0 = np.log(s0)/2*smask
    gs[0][0] += CSAL0[0]*x0 + csal1[0]
    if dtpmax > 0:
        gs[0][1] += (CSAL0[1]*x0 + csal1[1])/TRED

    # Scale everything by one more factor of salinity
    for dp in range(dtpmax+1):
        for dt in range(dtpmax+1-dp):
            gs[dp][dt] *= s1
    # Reformat elements to be floats
    if scalar:
        aux.makescalar(gs)
    return gs


def gibbs(salt, temp, pres, dtpmax, orig=False):
    """Calculate the Gibbs function for seawater.

    Calculate the Gibbs function for seawater from the absolute salinity,
    in-situ temperature, and seawater (gauge) pressure and its derivatives with
    respect to temperature and pressure.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        temp (float or array): In-situ temperature in degrees Celsius.
        pres (float or array): Seawater pressure in dbar (absolute pressure
            minus 1 atm).
        dtpmax (int >= 0): Maximum number of (t,p)-derivatives to take.
        orig (bool, optional): If True, the original coefficients for the Gibbs
            function are used. If False (default), then adjusted coefficients
            are used which satisfy that the entropy and enthalpy of seawater
            are 0 under standard conditions.

    Returns:
        gs (list of list of list of float or array): Gibbs free energy of
            seawater in units of
                (J kg-1) / (g kg-1)^ds / (deg C)^dt / (dbar)^dp.
            The derivatives are returned with those in temperature varying
            first followed by pressure. For dtpmax=3, this gives:
                gs = [[g, gt, gtt, gttt], [gp, gtp, gttp], [gpp, gtpp],
                      [gppp]].
    """
    gs = gibbs_sal(salt, temp, pres, dtpmax, orig=orig)
    gws = gibbs_wat(temp, pres, dtpmax, orig=orig)
    for dp in range(dtpmax+1):
        for dt in range(dtpmax+1-dp):
            gs[dp][dt] += gws[dp][dt]
    return gs


def gibbs0(salt, temp, dtpmax, orig=False):
    """Calculate the Gibbs function for seawater at 1 atm.

    Calculate the Gibbs function for seawater at 1 atm from the absolute
    salinity and in-situ temperature and its derivatives with respect to
    temperature and pressure.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        temp (float or array): In-situ temperature in degrees Celsius.
        dtpmax (int >= 0): Maximum number of (t,p)-derivatives to take.
        orig (bool, optional): If True, the original coefficients for the Gibbs
            function are used. If False (default), then adjusted coefficients
            are used which satisfy that the entropy and enthalpy of seawater
            are 0 under standard conditions.

    Returns:
        gs (list of list of list of float or array): Gibbs free energy of
            seawater in units of
                (J kg-1) / (g kg-1)^ds / (deg C)^dt / (dbar)^dp.
            The derivatives are returned with those in temperature varying
            first followed by pressure. For dtpmax=3, this gives:
                gs = [[g, gt, gtt, gttt], [gp, gtp, gttp], [gpp, gtpp],
                      [gppp]].
    """
    gs = gibbs_sal0(salt, temp, dtpmax, orig=orig)
    gws = gibbs_wat0(temp, dtpmax, orig=orig)
    for dp in range(dtpmax+1):
        for dt in range(dtpmax+1-dp):
            gs[dp][dt] += gws[dp][dt]
    return gs


# Check consistency of reference values
def checkwatref():
    """Check the consistency of the pure water reference values.

    The reference state of pure water is that the entropy and internal energy
    should be 0 at the triple point. This function calculates these values with
    both the original coefficients and the adjusted values to check how well
    these conditions are met.

    Arguments: None.

    Returns:
        entwtp_orig (float): Triple-point entropy in J kg-1 K-1 using the
            original coefficients.
        uwtp_orig (float): Triple-point internal energy in J kg-1 using the
            original coefficients.
        entwtp_adj, uwtp_adj (float): Triple-point entropy and internal energy
            using the adjusted coefficients.
    """
    # Get the triple-point Gibbs function values
    ttp = TTP - TCELS
    ptp = (PTP - PATM)/DBAR2PA
    gwtps_orig = gibbs_wat(ttp, ptp, 1, orig=True)
    gwtps_adj = gibbs_wat(ttp, ptp, 1, orig=False)

    # Calculate entropy and internal energy
    out = list()
    for gs in (gwtps_orig, gwtps_adj):
        ((g, gt), (gp,)) = gs
        ent = -gt
        u = g + TTP*ent - PTP/DBAR2PA*gp
        out += [ent, u]

    # Format results for output
    entwtp_orig, uwtp_orig, entwtp_adj, uwtp_adj = out[:]
    return (entwtp_orig, uwtp_orig, entwtp_adj, uwtp_adj)


def checksearef():
    """Check the consistency of the seawater reference values.

    The reference state of seawater is that the entropy and enthalpy should be
    0 at standard salinity, 0 Celsius, and 1 atm. This function calculates
    these values with both the original coefficients and the adjusted values to
    check how well these conditions are met.

    Arguments: None.

    Returns:
        ent0_orig (float): Standard entropy in J kg-1 K-1 using the original
            coefficients.
        h0_orig (float): Standard enthalpy in J kg-1 using the original
            coefficients.
        ent0_adj, h0_adj (float): Standard entropy and enthalpy using the
            adjusted coefficients.
    """
    # Get the standard Gibbs function values
    salt, temp = SAL0, 0.
    gs0_orig = gibbs0(salt, temp, 1, orig=True)
    gs0_adj = gibbs0(salt, temp, 1, orig=False)

    # Calculate entropy and enthalpy
    out = list()
    for gs in (gs0_orig, gs0_adj):
        ((g, gt), __) = gs
        ent = -gt
        h = g + TCELS*ent
        out += [ent, h]

    # Format results for output
    ent0_orig, h0_orig, ent0_adj, h0_adj = out[:]
    return (ent0_orig, h0_orig, ent0_adj, h0_adj)


# Temperature-potential temp conversion functions
def temp2pottemp(salt, temp, pres, tpot0=None, **rootkwargs):
    """Calculate temperature -> potential temperature.

    Calculate the potential temperature from the absolute salinity, in-situ
    temperature, and gauge (seawater) pressure. Applies either Newton's method
    or Halley's method. See `aux.rootfinder` for details on implementation and
    control arguments.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        temp (float or array): In-situ temperature in degrees Celsius.
        pres (float or array): Seawater pressure in dbar (absolute pressure
            minus 1 atm).
        tpot0 (float or array, optional): Initial estimate of potential
            temperature in degrees Celsius. If None (default) then the in-situ
            temperature is used.
        rootkwargs (dict, optional): Additional arguments for the root finder;
            see `aux.rootfinder` for available arguments and defaults.

    Returns:
        tpot (float or array): Potential temperature in degrees Celsius.
    """
    # Set initial guess
    if tpot0 is None:
        tpot0 = temp

    # Set up a function for the rootfinder
    update = rootkwargs.get('update', 'newton')
    if update == 'newton':
        dtpmax = 2
    elif update == 'halley':
        dtpmax = 3
    else:
        raise ValueError(
            'The update method must be either "newton" or "halley"')
    ((__, y0), __) = gibbs(salt, temp, pres, 1)
    args = (salt, pres)

    def derfun(tpot, salt, pres):
        gs = gibbs0(salt, tpot, dtpmax)
        return gs[0][1:]

    # Apply the root-finding method
    tpot = aux.rootfinder(derfun, y0, tpot0, TMIN, ENTMIN, args, **rootkwargs)
    return tpot


def pottemp2temp(salt, tpot, pres, temp0=None, **rootkwargs):
    """Calculate potential temperature -> temperature.

    Calculate the in-situ temperature from the absolute salinity, potential
    temperature, and gauge (seawater) pressure. Applies either Newton's method
    or Halley's method. See `aux.rootfinder` for details on implementation and
    control arguments.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        tpot (float or array): Potential temperature in degrees Celsius.
        pres (float or array): Seawater pressure in dbar (absolute pressure
            minus 1 atm).
        temp0 (float or array, optional): Initial estimate of in-situ
            temperature in degrees Celsius. If None (default) then the
            potential temperature is used.
        rootkwargs (dict, optional): Additional arguments for the root finder;
            see `aux.rootfinder` for available arguments and defaults.

    Returns:
        temp (float or array): In-situ temperature in degrees Celsius.
    """
    # Set initial guess
    if temp0 is None:
        temp0 = tpot

    # Set up a function for the rootfinder
    update = rootkwargs.get('update', 'newton')
    if update == 'newton':
        dtpmax = 2
    elif update == 'halley':
        dtpmax = 3
    else:
        raise ValueError(
            'The update method must be either "newton" or "halley"')
    ((__, y0), __) = gibbs0(salt, tpot, 1)
    args = (salt, pres)

    def derfun(temp, salt, pres):
        gs = gibbs(salt, temp, pres, dtpmax)
        return gs[0][1:]

    # Apply the root-finding method
    temp = aux.rootfinder(derfun, y0, temp0, TMIN, ENTMIN, args, **rootkwargs)
    return temp


# Potential-conservative temperature conversion
def pottemp2contemp(salt, tpot):
    """Calculate conservative temperature from potential temperature.

    Calculate the conservative temperature (scaled potential enthalpy) from the
    absolute salinity and potential temperature.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        tpot (float or array): Potential temperature in degrees Celsius.

    Returns:
        tcon (float or array): Conservative temperature in degrees Celsius.
    """
    # Get the Gibbs function values *with the adjusted coefficients*
    ((g, gt), __) = gibbs0(salt, tpot, 1, orig=False)

    # Calculate enthalpy
    h = g - (TCELS+tpot)*gt
    tcon = h/CSEA
    return tcon


def contemp2pottemp(salt, tcon, tpot0=None, **rootkwargs):
    """Calculate conservative temp -> potential temp.

    Calculate the potential temperature from the absolute salinity and
    conservative temperature. Applies either Newton's method or Halley's
    method. See `aux.rootfinder` for details on implementation and control
    arguments.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        tcon (float or array): Conservative temperature in degrees Celsius.
        tpot0 (float or array, optional): Initial estimate of potential
            temperature in degrees Celsius. If None (default) then the
            conservative temperature is used.
        rootkwargs (dict, optional): Additional arguments for the root finder;
            see `aux.rootfinder` for available arguments and defaults.

    Returns:
        tpot (float or array): Potential temperature in degrees Celsius.
    """
    # Set initial guess
    if tpot0 is None:
        tpot0 = tcon

    # Set up a function for the rootfinder
    update = rootkwargs.get('update', 'newton')
    if update == 'newton':
        dtpmax = 2
    elif update == 'halley':
        dtpmax = 3
    else:
        raise ValueError(
            'The update method must be either "newton" or "halley"')
    y0 = CSEA*tcon
    args = (salt,)

    def derfun(tpot, salt):
        # Calculate Gibbs function *with adjusted coefficients*
        (g0s, *__) = gibbs0(salt, tpot, dtpmax, orig=False)
        tabs = TCELS + tpot
        hs = [g0s[0]-tabs*g0s[1], -tabs*g0s[2]]
        if dtpmax > 2:
            hs.append(-g0s[2] - tabs*g0s[3])
        return hs

    # Apply the root-finding method
    tpot = aux.rootfinder(
        derfun, y0, tpot0, TMIN, CSEA*TMIN, args, **rootkwargs)
    return tpot


# Temperature-conservative temperature conversion
def temp2contemp(salt, temp, pres, returntpot=False, tpot0=None, **rootkwargs):
    """Calculate temperature -> conservative temperature.

    Calculate the conservative temperature from the absolute salinity, in-situ
    temperature, and gauge (seawater) pressure. First calculates the potential
    temperature and then the conservative temperature; the potential
    temperature can be returned as well.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        temp (float or array): In-situ temperature in degrees Celsius.
        pres (float or array): Seawater pressure in dbar (absolute pressure
            minus 1 atm).
        returntpot (bool, optional): If True (default False) then the potential
            temperature is returned as well.
        tpot0 (float or array, optional): Initial estimate of potential
            temperature in degrees Celsius. If None (default) then the in-situ
            temperature is used.
        rootkwargs (dict, optional): Additional arguments for the root finder;
            see `aux.rootfinder` for available arguments and defaults.

    Returns:
        tcon (float or array): Conservative temperature in degrees Celsius.
        tpot (float or array, optional): Potential temperature in degrees
            Celsius. Returned if `returntpot` is True.
    """
    tpot = temp2pottemp(salt, temp, pres, tpot0=tpot0, **rootkwargs)
    tcon = pottemp2contemp(salt, tpot)
    if returntpot:
        return (tcon, tpot)
    else:
        return tcon


def contemp2temp(
        salt, tcon, pres, returntpot=False, tpot0=None, temp0=None,
        **rootkwargs):
    """Calculate conservative temperature -> temperature.

    Calculate the in-situ temperature from the absolute salinity, conservative
    temperature, and gauge (seawater) pressure. First calculates the potential
    temperature and then the in-situ temperature; the potential temperature can
    be returned as well.

    Arguments:
        salt (float or array): Absolute salinity in g kg-1.
        tcon (float or array): Conservative temperature in degrees Celsius.
        pres (float or array): Seawater pressure in dbar (absolute pressure
            minus 1 atm).
        returntpot (bool, optional): If True (default False) then the potential
            temperature is returned as well.
        tpot0 (float or array, optional): Initial estimate of potential
            temperature in degrees Celsius. If None (default) then the
            conservative temperature is used.
        temp0 (float or array, optional): Initial estimate of the in-situ
            temperature in degrees Celsius. If None (default) then the
            potential temperature is used.
        rootkwargs (dict, optional): Additional arguments for the root finder;
            see `aux.rootfinder` for available arguments and defaults.

    Returns:
        temp (float or array): In-situ temperature in degrees Celsius.
        tpot (float or array, optional): Potential temperature in degrees
            Celsius. Returned if `returntpot` is True.
    """
    tpot = contemp2pottemp(salt, tcon, tpot0=tpot0, **rootkwargs)
    temp = pottemp2temp(salt, tpot, pres, temp0=temp0, **rootkwargs)
    if returntpot:
        return (temp, tpot)
    else:
        return temp
