#!/usr/bin/env python3
"""Constants in the various equation of state routines.
"""

# Physical constants
TCELS = 273.15  # 0 Celsius in K
PATM = 101325.  # 1 atm in Pa
GRAV = 9.80665  # Standard gravity, m s-2
TTP = 273.16  # Triple point temperature in K
PTP = 611.657  # Triple point pressure in Pa
DBAR2PA = 1e4  # Conversion factor, Pa/dbar

# Reference values
CSEA = 3991.86795711963  # Standard heat capacity of seawater, J kg-1 K-1
SAL0 = 35.16504  # Salinity of KCl-normalized seawater, g kg-1
SRED = SAL0 * 40./35.  # Reducing salinity, g kg-1
TRED = 40.  # Reducing temperature, deg C
ZRED = 1e4  # Reducing depth, m
PRED = 1e4  # Reducing pressure, dbar
SNDFACTOR = (PRED*DBAR2PA)**.5  # Coefficient in speed of sound, Pa^(.5)
RHOBSQ = 1020.  # Reference density for depth-pressure conversion, kg/m3

# Numerical constants
DTASBSQ = 32.  # Salinity offset for Boussinesq formulation, g kg-1
DTASCMP = 24.  # Salinity offset for compressile formulation, g kg-1
SMIN = 1e-8  # Minimum salinity for calculating logarithm, g kg-1
TMIN = 1e-2  # Minimum temperature in relative tolerance, deg C
ENTMIN = 1e-8  # Minimum entropy in relative tolerance, J kg-1 K-1

# File constants
TXTFILE = 'coefs.txt'  # Plain-text file of coefficients
NPZFILE = 'coefs.npz'  # Numpy array save file
