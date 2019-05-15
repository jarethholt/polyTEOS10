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

# Physical constants
TCELS = 273.15  # 0 Celsius in K
GRAV = 9.80665  # Standard gravity, m s-2
CSEA = 3991.86795711963  # Standard heat capacity of seawater, J kg-1 K-1
SNORM = 35.16504  # Salinity of KCl-normalized seawater, g kg-1
SRED = SNORM * 40./35.  # Reducing salinity, g kg-1
CTREF = 40.  # Reducing temperature, deg C
ZREF = 1e4  # Reducing depth, m
PREF = 1e4  # Reducing pressure, dbar
DBAR2PA = 1e4  # Conversion factor, Pa/dbar
SNDFACTOR = (PREF*DBAR2PA)**.5  # Coefficient in speed of sound, Pa(.5)


## Boussinesq form constants
RHOBSQ = 1020.  # Reference density for depth-pressure conversion, kg/m3
DTASBSQ = 32.  # Salinity offset, g kg-1

# Coefficients for the vertical reference profile of density
BSQCOEFS0 = (0.,
     4.6494977072e+01,-5.2099962525e+00, 2.2601900708e-01, 6.4326772569e-02,
     1.5616995503e-02,-1.7243708991e-03)

# Coefficients for the density anomaly
NSTBSQ = (6, 4, 2, 1)
BSQCOEFS1 = (
     8.0189615746e+02, 8.6672408165e+02,-1.7864682637e+03, 2.0375295546e+03,
    -1.2849161071e+03, 4.3227585684e+02,-6.0579916612e+01, 2.6010145068e+01,
    -6.5281885265e+01, 8.1770425108e+01,-5.6888046321e+01, 1.7681814114e+01,
    -1.9193502195e+00,-3.7074170417e+01, 6.1548258127e+01,-6.0362551501e+01,
     2.9130021253e+01,-5.4723692739e+00, 2.1661789529e+01,-3.3449108469e+01,
     1.9717078466e+01,-3.1742946532e+00,-8.3627885467e+00, 1.1311538584e+01,
    -5.3563304045e+00, 5.4048723791e-01, 4.8169980163e-01,-1.9083568888e-01,
     1.9681925209e+01,-4.2549998214e+01, 5.0774768218e+01,-3.0938076334e+01,
     6.6051753097e+00,-1.3336301113e+01,-4.4870114575e+00, 5.0042598061e+00,
    -6.5399043664e-01, 6.7080479603e+00, 3.5063081279e+00,-1.8795372996e+00,
    -2.4649669534e+00,-5.5077101279e-01, 5.5927935970e-01, 2.0660924175e+00,
    -4.9527603989e+00, 2.5019633244e+00, 2.0564311499e+00,-2.1311365518e-01,
    -1.2419983026e+00,-2.3342758797e-02,-1.8507636718e-02, 3.7969820455e-01
)


## Stiffened Boussinesq from constants
# Coefficients for the vertical reference scale
STIFCOEFS0 = (1.,
     4.5238001132e-02,-5.0691457704e-03, 2.1990865986e-04, 6.2587720090e-05,
     1.5194795322e-05,-1.6777531159e-06)

# Coefficients for the normalized density
STIFCOEFS1 = (
     8.0185969881e+02, 8.6694399997e+02,-1.7869886805e+03, 2.0381548497e+03,
    -1.2853207957e+03, 4.3240996619e+02,-6.0597695001e+01, 2.6018938392e+01,
    -6.5349779146e+01, 8.1938301569e+01,-5.7075042739e+01, 1.7778970855e+01,
    -1.9385269480e+00,-3.7047586837e+01, 6.1469677558e+01,-6.0273564480e+01,
     2.9086147388e+01,-5.4641145446e+00, 2.1645370860e+01,-3.3415215649e+01,
     1.9694119706e+01,-3.1710494147e+00,-8.3587258634e+00, 1.1301873278e+01,
    -5.3494903247e+00, 5.4258499460e-01, 4.7964098705e-01,-1.9098981559e-01,
     2.1989266031e+01,-4.2043785414e+01, 4.8565183521e+01,-3.0473875108e+01,
     6.5025796369e+00,-1.3731593003e+01,-4.3667263842e+00, 5.2899298884e+00,
    -7.1323826203e-01, 7.4843325711e+00, 3.1442996192e+00,-1.8141771987e+00,
    -2.6010182316e+00,-4.9866739215e-01, 5.5882364387e-01, 1.1144125393e+00,
    -4.5413502768e+00, 2.7242121539e+00, 2.8508446713e+00,-4.4471361300e-01,
    -1.5059302816e+00, 1.9817079368e-01,-1.7905369937e-01, 2.5254165600e-01)


## 55-term compressible EOS constants
DTASCMP = 24.  # Salinity offset, g kg-1

# Coefficients for the vertical reference volume profile
CMPCOEFS0 = (0.,
    -4.4015007269e-05, 6.9232335784e-06,-7.5004675975e-07, 1.7009109288e-08,
    -1.6884162004e-08, 1.9613503930e-09)

# Coefficients for the volume anomaly
C55COEFS1 = (
     1.0772899069e-03,-3.1263658781e-04, 6.7615860683e-04,-8.6127884515e-04,
     5.9010812596e-04,-2.1503943538e-04, 3.2678954455e-05,-1.4949652640e-05,
     3.1866349188e-05,-3.8070687610e-05, 2.9818473563e-05,-1.0011321965e-05,
     1.0751931163e-06, 2.7546851539e-05,-3.6597334199e-05, 3.4489154625e-05,
    -1.7663254122e-05, 3.5965131935e-06,-1.6506828994e-05, 2.4412359055e-05,
    -1.4606740723e-05, 2.3293406656e-06, 6.7896174634e-06,-8.7951832993e-06,
     4.4249040774e-06,-7.2535743349e-07,-3.4680559205e-07, 1.9041365570e-07,
    -1.6889436589e-05, 2.1106556158e-05,-2.1322804368e-05, 1.7347655458e-05,
    -4.3209400767e-06, 1.5355844621e-05, 2.0914122241e-06,-5.7751479725e-06,
     1.0767234341e-06,-9.6659393016e-06,-7.0686982208e-07, 1.4488066593e-06,
     3.1134283336e-06, 7.9562529879e-08,-5.6590253863e-07, 1.0500241168e-06,
     1.9600661704e-06,-2.1666693382e-06,-3.8541359685e-06, 1.0157632247e-06,
     1.7178343158e-06,-4.1503454190e-07, 3.5627020989e-07,-1.1293871415e-07)


## 75-term compressible EOS constants
# Coefficients for the vertical reference volume profile
CMPCOEFS0 = (0.,
    -4.4015007269e-05, 6.9232335784e-06,-7.5004675975e-07, 1.7009109288e-08,
    -1.6884162004e-08, 1.9613503930e-09)

# Coefficients for the volume anomaly
NSTC75 = (6, 5, 4, 2, 1, 0)
C75COEFS1 = (
     1.0769995862e-03,-3.1038981976e-04, 6.6928067038e-04,-8.5047933937e-04,
     5.8086069943e-04,-2.1092370507e-04, 3.1932457305e-05,-1.5649734675e-05,
     3.5009599764e-05,-4.3592678561e-05, 3.4532461828e-05,-1.1959409788e-05,
     1.3864594581e-06, 2.7762106484e-05,-3.7435842344e-05, 3.5907822760e-05,
    -1.8698584187e-05, 3.8595339244e-06,-1.6521159259e-05, 2.4141479483e-05,
    -1.4353633048e-05, 2.2863324556e-06, 6.9111322702e-06,-8.7595873154e-06,
     4.3703680598e-06,-8.0539615540e-07,-3.3052758900e-07, 2.0543094268e-07,
    -1.6784136540e-05, 2.4262468747e-05,-3.4792460974e-05, 3.7470777305e-05,
    -1.7322218612e-05, 3.0927427253e-06, 1.8505765429e-05,-9.5677088156e-06,
     1.1100834765e-05,-9.8447117844e-06, 2.5909225260e-06,-1.1716606853e-05,
    -2.3678308361e-07, 2.9283346295e-06,-4.8826139200e-07, 7.9279656173e-06,
    -3.4558773655e-06, 3.1655306078e-07,-3.4102187482e-06, 1.2956717783e-06,
     5.0736766814e-07, 3.0623833435e-06,-5.8484432984e-07,-4.8122251597e-06,
     4.9263106998e-06,-1.7811974727e-06,-1.1736386731e-06,-5.5699154557e-06,
     5.4620748834e-06,-1.3544185627e-06, 2.1305028740e-06, 3.9137387080e-07,
    -6.5731104067e-07,-4.6132540037e-07, 7.7618888092e-09,-6.3352916514e-08,
    -3.8088938393e-07, 3.6310188515e-07, 1.6746303780e-08,-3.6527006553e-07,
    -2.7295696237e-07, 2.8695905159e-07, 8.8302421514e-08,-1.1147125423e-07,
     3.1454099902e-07, 4.2369007180e-09)


## Helper functions
def poly1d(z,coefs):
    """Evaluate a 1d polynomial with derivative.
    
    Evaluate a polynomial in one variable with the given coefficients at the
    given point as well as its first derivative. The point can be given as a
    numpy array, in which case the outputs have the same type and shape as that
    array.
    
    Parameters
    ----------
    z     : float or array_like
            Point(s) to evaluate the polynomial at.
    coefs : iterable of float
            Coefficients of the polynomial, from lowest degree to highest.
    
    Returns
    -------
    p    : Value of the polynomial at the point(s) `z`.
    dpdz : Value of the derivative at the point(s) `z`.
    """
    p, dpdz = 0., 0.
    for coef in coefs[-1::-1]:
        dpdz = dpdz*z + p
        p = p*z + coef
    return (p, dpdz)

def poly3d(x,y,z,coefs,nxys):
    """Evaluate a 3d polynomial with derivatives.
    
    Evaluate a polynomial in three variables with the given coefficients at the
    given point, as well as its first derivatives with respect to each variable.
    The points can be given as numpy arrays as long as they are broadcastable
    against each other; all outputs will have this shape and type.
    
    The polynomial coefficients are given as a flat iterable (tuple, list, or
    array). They should be ordered from lowest degree to highest, with the
    degree in `x` changing first, followed by the degrees for `y` and `z`. The
    degrees might therefore be:
        000 100 200 300 010 110 210 ... 002 102 012.
    It is assumed that for each degree `nz` of `z`, there are coefficients for
    every degree of (`x`,`y`) up to the number `nxys[nz]`. In the above
    example, `nxys[0]=3` and `nxys[2]=1`.
    
    Parameters
    ----------
    x, y, z : float or array_like
              Values of the three variables at which to evaluate the polynomial.
    coefs   : iterable of float
              Coefficients of the polynomial.
    nxys    : iterable of int
              Maximum degree of `(x,y)` for every order of `z`.
    
    Returns
    -------
    p             : Value of the polynomial at `(x,y,z)`.
    dpx, dpy, dpz : Derivative of the polynomial with respect to `(x,y,z)`.
    """
    p, dpx, dpy, dpz = 0., 0., 0., 0.
    ind = -1
    for nxy in nxys[-1::-1]:
        p2, dp2x, dp2y = 0., 0., 0.
        for iy in range(nxy,-1,-1):
            p3, dp3x = 0., 0.
            for ix in range(nxy-iy,-1,-1):
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


## Boussinesq functions
def eos_bsq(salt,temp,dpth):
    """Calculate Boussinesq density.
    
    Calculate the density and related quantities using a Boussinesq
    approximation from the salinity, temperature, and pressure. In the
    Boussinesq form, any term divided by the full density should use the
    reference density RHOBSQ (1020 kg m-3).
    
    Any of the inputs can be numpy arrays as well as floats. If any are arrays,
    then they must be broadcastable against each other. All outputs will then
    have the broadcast shape.
    
    Parameters
    ----------
    salt : Absolute salinity in g kg-1.
    temp : Conservative temperature in degrees Celsius.
    dpth : Seawater depth in m; equivalently, seawater reference hydrostatic
           pressure in dbar.
    
    Returns
    -------
    rho  : Density in kg m-3.
    absq : Modified thermal expansion coefficient (-drho/dtemp) in kg m-3 K-1.
    bbsq : Modified haline contraction coefficient (drho/dsalt) in
           kg m-3 (g kg-1)-1.
    csnd : Speed of sound in m s-1.
    
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
    tau = temp / CTREF
    zet = dpth / ZREF
    
    # Vertical reference profile of density
    r0, dr0z = poly1d(zet,BSQCOEFS0)
    
    # Density anomaly
    r1, dr1s, dr1t, dr1z = poly3d(sig,tau,zet,BSQCOEFS1,NSTBSQ)
    
    # Calculate physically-relevant quantities
    rho = r0 + r1
    absq = -dr1t/CTREF
    bbsq = dr1s/(2*sig*SRED)
    csnd = (RHOBSQ*GRAV*ZREF/(dr0z + dr1z))**.5
    return (rho, absq, bbsq, csnd)

def eos_stif(salt,temp,dpth):
    """Calculate stiffened density.
    
    Calculate the density and related quantities using a stiffened Boussinesq
    approximation from the salinity, temperature, and pressure. In the
    Boussinesq form, any term divided by the full density should use the
    reference density RHOBSQ (1020 kg m-3).
    
    Any of the inputs can be numpy arrays as well as floats. If any are arrays,
    then they must be broadcastable against each other. All outputs will then
    have the broadcast shape.
    
    Parameters
    ----------
    salt : Absolute salinity in g kg-1.
    temp : Conservative temperature in degrees Celsius.
    dpth : Seawater depth in m; equivalently, seawater reference hydrostatic
           pressure in dbar.
    
    Returns
    -------
    rho  : Density in kg m-3.
    absq : Modified thermal expansion coefficient (-drho/dtemp) in kg m-3 K-1.
    bbsq : Modified haline contraction coefficient (drho/dsalt) in
           kg m-3 (g kg-1)-1.
    csnd : Speed of sound in m s-1.
    
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
    tau = temp / CTREF
    zet = dpth / ZREF
    
    # Vertical reference profile of density
    r1, dr1z = poly1d(zet,STIFCOEFS0)
    
    # Normalized density
    rdot, drdots, drdott, drdotz = poly3d(sig,tau,zet,STIFCOEFS1,NSTBSQ)
    
    # Return all physical quantities
    rho = r1 * rdot
    absq = -r1/CTREF * drdott
    bbsq = r1/(2*sig*SRED) * drdots
    csnd = (RHOBSQ*GRAV*ZREF/(rdot*dr1z + r1*drdotz))**.5
    return (rho, absq, bbsq, csnd)

def calnsq_bsq(absq,bbsq,dctdz,dsadz):
    """Calculate the Boussinesq stratification.
    
    Calculate the square of the buoyancy frequency for a Boussinesq system,
    i.e. the stratification. Here, absq and bbsq are the modified thermal
    expansion and haline contraction coefficients returned by either eos_bsq or
    eos_stif, which are both Boussinesq equations of state.
    
    Any of the inputs can be numpy arrays as well as floats. If any are arrays,
    then they must be broadcastable against each other. In particular, the
    expansion and contraction coefficients have to be on the same vertical grid
    as the temperature and salinity gradients.
    
    Parameters
    ----------
    absq  : Modified thermal expansion coefficient (-drho/dtemp) in kg m-3 K-1.
    bbsq  : Modified haline contraction coefficient (drho/dsalt) in
            kg m-3 (g kg-1)-1.
    dctdz : Vertical gradient of the conservative temperature in K m-1.
    dsadz : Vertical gradient of the absolute salinity in g kg-1 m-1.
    
    Returns
    -------
    nsq : Squared buoyancy frequency in s-2.
    
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


## Compressible functions
def eos_c55(salt,temp,pres):
    """Calculate specific volume (55-term polynomial).
    
    Calculate the specific volume and related quantities using a 55-term
    polynomial fit. This equation of state is compressible and thus uses
    pressure rather than depth.
    
    Any of the inputs can be numpy arrays as well as floats. If any are arrays,
    then they must be broadcastable against each other. All outputs will then
    have the broadcast shape.
    
    Parameters
    ----------
    salt : Absolute salinity in g kg-1.
    temp : Conservative temperature in degrees Celsius.
    pres : Seawater pressure in dbar (absolute pressure minus 1 atm).
    
    Returns
    -------
    svol  : Specific volume (1/density) in m3 kg-1.
    alpha : Thermal expansion coefficient K-1.
    beta  : Haline contraction coefficient in (g kg-1)-1.
    csnd  : Speed of sound in m s-1.
    
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
    tau = temp / CTREF
    phi = pres / PREF
    
    # Vertical reference profile of density
    v0, dv0p = poly1d(phi,CMPCOEFS0)
    
    # Density anomaly
    dta, ddtas, ddtat, ddtap = poly3d(sig,tau,phi,C55COEFS1,NSTBSQ)
    
    # Return all physical quantities
    svol = v0 + dta
    alpha = ddtat/CTREF / svol
    beta = -ddtas/(2*sig*SRED) / svol
    csnd = svol * (-(dv0p + ddtap))**(-.5) * SNDFACTOR
    return (svol, alpha, beta, csnd)

def eos_c75(salt,temp,pres):
    """Calculate specific volume (75-term polynomial).
    
    Calculate the specific volume and related quantities using a 75-term
    polynomial fit. This equation of state is compressible and thus uses
    pressure rather than depth.
    
    Any of the inputs can be numpy arrays as well as floats. If any are arrays,
    then they must be broadcastable against each other. All outputs will then
    have the broadcast shape.
    
    Parameters
    ----------
    salt : Absolute salinity in g kg-1.
    temp : Conservative temperature in degrees Celsius.
    pres : Seawater pressure in dbar (absolute pressure minus 1 atm).
    
    Returns
    -------
    svol  : Specific volume (1/density) in m3 kg-1.
    alpha : Thermal expansion coefficient K-1.
    beta  : Haline contraction coefficient in (g kg-1)-1.
    csnd  : Speed of sound in m s-1.
    
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
    tau = temp / CTREF
    phi = pres / PREF
    
    # Vertical reference profile of density
    v0, dv0p = poly1d(phi,CMPCOEFS0)
    
    # Density anomaly
    dta, ddtas, ddtat, ddtap = poly3d(sig,tau,phi,C75COEFS1,NSTC75)
    
    # Return all physical quantities
    svol = v0 + dta
    alpha = ddtat/CTREF / svol
    beta = -ddtas/(2*sig*SRED) / svol
    csnd = svol * (-(dv0p + ddtap))**(-.5) * SNDFACTOR
    return (svol, alpha, beta, csnd)


## Main script: Run doctest
if __name__ == '__main__':
    import doctest
    doctest.testmod()

