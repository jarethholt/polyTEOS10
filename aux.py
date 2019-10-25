#!/usr/bin/env python3
"""Module for mathematical functions.

This module isolates some of the mathematical functions used by the various
equation of state routines. In particular, it provides several functions for
evaluating polynomials in 1, 2, and 3 variables, including derivatives. It also
provides derivative-based root-finding methods (Newton and Halley) in a generic
way.
"""

# Import statements
import numpy as np
import warnings

# Constants
MAXITER = 10  # Suggested maximum number of iterations in root-finder
RTOL = 1e-8  # Suggested relative tolerance


# Scalar functions
def isscalar(*args):
    """Determine whether a collection of objects does not need broadcasting.

    Determine whether a number of objects (each either a float or a numpy
    array) need to be broadcast against each other. If not, the return type
    should be a scalar (float) instead of an array. This function returns
    whether or not the result is scalar.

    Arguments:
        arg0, arg1, ... (float or array): The elements to be tested.

    Returns:
        scalar (bool): True if all elements of the sequence are floats.
    """
    scalar = (not any(isinstance(arg, np.ndarray) for arg in args))
    return scalar


def makescalar(origlist):
    """Recursive function to make all elements of nested lists into floats.

    To accommodate array inputs, all the functions here first broadcast to
    numpy arrays, generating nested lists of arrays. When the inputs are
    floats, these 0-d arrays should be cast back into floats. This function
    acts recursively on nested lists to do this.

    Arguments:
        origlist (list): List of elements to be converted. Elements are either
            numpy arrays or another list. These lists are modified in-place.

    Returns None.
    """
    for (i, elem) in enumerate(origlist):
        if isinstance(elem, list):
            makescalar(origlist[i])
        else:
            origlist[i] = float(elem)
    return None


# Polynomials
def poly1d(z, coefs):
    """Evaluate a 1d polynomial.

    Evaluate a polynomial in one variable with the given coefficients at the
    given point. Uses Horner's method to speed up calculation.

    Arguments:
        z (float or array): Point(s) to evaluate the polynomial at.
        coefs (iterable of float): Coefficients of the polynomial, from lowest
            degree to highest.

    Returns:
        p (float or array): Value of the polynomial at the point(s) `z`.
    """
    # Track whether the output should be scalar or array
    scalar = isscalar(z, coefs[0])
    out = np.broadcast(z, coefs[0])
    p = np.zeros(out.shape)

    # Iterate backwards over the coefficients
    for coef in coefs[-1::-1]:
        p *= z
        p += coef

    # Reformat for scalar output
    if scalar:
        p = float(p)
    return p


def poly1d_1der(z, coefs, zscale=1.):
    """Evaluate a 1d polynomial with derivative.

    Evaluate a polynomial in one variable with the given coefficients at the
    given point as well as its first derivative. The point can be given as a
    numpy array, in which case the outputs have the same type and shape as that
    array.

    Arguments:
        z (float or array): Point(s) to evaluate the polynomial at.
        coefs (iterable of float): Coefficients of the polynomial, from lowest
            degree to highest.
        zscale (float, optional): Amount to scale derivatives by (default 1.).
            Useful when calculating physical derivatives.

    Returns:
        p (float or array): Value of the polynomial at the point(s) `z`.
        dpdz (float or array): Value of the derivative at the point(s) `z`.
    """
    # Track whether the output should be scalar or array
    scalar = isscalar(z, coefs[0])
    out = np.broadcast(z, coefs[0])
    p, dp = [np.zeros(out.shape) for __ in range(2)]

    # Loop backwards through coefficients
    for coef in coefs[-1::-1]:
        dp *= z
        dp += p
        p *= z
        p += coef

    # Reformat for scalar output
    dp /= zscale
    if scalar:
        ps = [p, dp]
        makescalar(ps)
        p, dp = ps[:]
    return (p, dp)


def poly1d_ders(z, coefs, dmax, zscale=1.):
    """Evaluate a 1d polynomial with multiple derivatives.

    Evaluate a polynomial in one variable with the given coefficients at the
    given points and its derivatives up to a maximum order.

    The points can be given as numpy arrays as long as they can be broadcast
    against each other; all outputs will have this shape and type.

    Arguments:
        z (float or array): Value(s) of each variable at which to evaluate the
            polynomial.
        coefs (iterable of float): Coefficients of the polynomial.
        dmax (int >= 0): Highest derivative to take with respect to z.
        zscale (float, optional): Amount to scale derivatives by (default 1.).
            Useful when calculating physical derivatives.

    Returns:
        p (length (dmax+1) list of float or array): Value of the polynomial
            and its derivatives at the given point(s).
    """
    # Track whether the output should be scalar or array
    scalar = isscalar(z, coefs[0])
    out = np.broadcast(z, coefs[0])
    ps = [np.zeros(out.shape) for dz in range(dmax+1)]

    # Loop backwards through coefficients
    for coef in coefs[-1::-1]:
        for i in range(dmax, 0, -1):
            ps[i] *= z
            ps[i] += i*ps[i-1]/zscale
        ps[0] *= z
        ps[0] += coef

    # Reformat for scalar output
    if scalar:
        makescalar(ps)
    return ps


def poly2d(y, z, coefs, jmaxs):
    """Evaluate a 2d polynomial.

    Evaluate a polynomial in two variables with the given coefficients at the
    given points.

    The points can be given as numpy arrays as long as they can be broadcast
    against each other; all outputs will have this shape and type.

    The polynomial coefficients are given as a flat iterable (tuple, list, or
    array). They should be ordered from lowest degree to highest, with the
    degree in `y` changing first, followed by the degree for `z`. The input
    `jmaxs` gives the number of degrees of `y` (j) for each degree of `z` (k).

    Arguments:
        y, z (float or array): Value(s) of each variable at which to evaluate
            the polynomial.
        coefs (iterable of float): Coefficients of the polynomial.
        jmaxs (iterable of int): Maximum degree in `y` for each degree of `z`.

    Returns:
        p (float or array): Value of the polynomial at the given point(s).
    """
    # Track whether the output should be scalar or array
    scalar = isscalar(y, z, coefs[0])
    out = np.broadcast(y, z, coefs[0])
    p, p2 = [np.zeros(out.shape) for __ in range(2)]

    # Loop backwards through coefficients in z then y
    ind = -1
    for jmax in jmaxs[-1::-1]:
        p2[...] = 0.
        for j in range(jmax, -1, -1):
            p2 *= y
            p2 += coefs[ind]
            ind -= 1
        p *= z
        p += p2

    # Reformat for scalar output
    if scalar:
        p = float(p)
    return p


def poly2d_1der(y, z, coefs, jmaxs, yscale=1., zscale=1.):
    """Evaluate a 2d polynomial with first derivatives.

    Evaluate a polynomial in two variables with the given coefficients at the
    given points, as well as the first derivatives with respect to each
    variable.

    The points can be given as numpy arrays as long as they can be broadcast
    against each other; all outputs will have this shape and type.

    The polynomial coefficients are given as a flat iterable (tuple, list, or
    array). They should be ordered from lowest degree to highest, with the
    degree in `y` changing first, followed by the degree for `z`. The input
    `jmaxs` gives the number of degrees of `y` (j) for each degree of `z` (k).

    Arguments:
        y, z (float or array): Value(s) of each variable at which to evaluate
            the polynomial.
        coefs (iterable of float): Coefficients of the polynomial.
        jmaxs (iterable of int): Maximum degree in `y` for each degree of `z`.
        yscale, zscale (float, optional): Amount to scale each derivative by
            (default 1.). Useful for calculating physical derivatives.

    Returns:
        p (float or array): Value of the polynomial at the given point(s).
        py, pz (float or array): Derivative of the polynomial with respect to
            each variable.
    """
    # Track whether the output should be scalar or array
    scalar = isscalar(y, z, coefs[0])
    out = np.broadcast(y, z, coefs[0])
    p, py, pz, p2, py2 = [np.zeros(out.shape) for __ in range(5)]

    # Loop backwards through coefficients in z then y
    ind = -1
    for jmax in jmaxs[-1::-1]:
        p2[...] = 0.
        py2[...] = 0.
        for j in range(jmax, -1, -1):
            py2 *= y
            py2 += p2
            p2 *= y
            p2 += coefs[ind]
            ind -= 1
        pz *= z
        pz += p
        p *= z
        p += p2
        py *= z
        py += py2

    # Reformat for scalar output
    py /= yscale
    pz /= zscale
    if scalar:
        ps = [p, py, pz]
        makescalar(ps)
        p, py, pz = ps[:]
    return (p, py, pz)


def poly2d_ders(y, z, coefs, jmaxs, dmax, yscale=1., zscale=1.):
    """Evaluate a 2d polynomial with derivatives.

    Evaluate a polynomial in two variables with the given coefficients at the
    given points, as well as the derivatives up to a specified maximum.

    The points can be given as numpy arrays as long as they can be broadcast
    against each other; all outputs will have this shape and type.

    The polynomial coefficients are given as a flat iterable (tuple, list, or
    array). They should be ordered from lowest degree to highest, with the
    degree in `y` changing first, followed by the degree for `z`. The input
    `jmaxs` gives the number of degrees of `y` (j) for each degree of `z` (k).

    Arguments:
        y, z (float or array): Value(s) of each variable at which to evaluate
            the polynomial.
        coefs (iterable of float): Coefficients of the polynomial.
        jmaxs (iterable of int): Maximum degree in `y` for each degree of `z`.
        dmax (int >= 0): Maximum number of derivatives to take.
        yscale, zscale (float, optional): Amount to scale each derivative by
            (default 1.). Useful for calculating physical derivatives.
        zscale (float, optional): Amount to scale z-derivatives by.

    Returns:
        ps (list of list of float or array): Value of the polynomial and its
            derivatives. The derivatives are returned with the y-derivative
            varying first followed by the z-derivative. For dmax=3, the
            structure returned is
                ps = [[p, py, pyy, pyyy], [pz, pyz, pyyz],
                      [pzz, pyzz], [pzzz]].
    """
    # Track whether the output should be scalar or array
    scalar = isscalar(y, z, coefs[0])
    out = np.broadcast(y, z, coefs[0])
    ps = [[np.zeros(out.shape) for dy in range(dmax+1-dz)]
          for dz in range(dmax+1)]
    ps2 = [np.zeros(out.shape) for dy in range(dmax+1)]

    # Loop backwards through coefficients in z then y
    ind = -1
    for jmax in jmaxs[-1::-1]:
        # Track values with only y-derivatives
        for dy in range(dmax+1):
            ps2[dy][...] = 0.
        for j in range(jmax, -1, -1):
            for dy in range(dmax, 0, -1):
                ps2[dy] *= y
                ps2[dy] += dy*ps2[dy-1]/yscale
            ps2[0] *= y
            ps2[0] += coefs[ind]
            ind -= 1

        # Calculate the z-derivative terms
        for dy in range(dmax+1):
            for dz in range(dmax-dy, 0, -1):
                ps[dz][dy] *= z
                ps[dz][dy] += dz*ps[dz-1][dy]/zscale
            ps[0][dy] *= z
            ps[0][dy] += ps2[dy]

    # Reformat for scalar output
    if scalar:
        makescalar(ps)
    return ps


def poly3d(x, y, z, coefs, ijmaxs):
    """Evaluate a 3d polynomial.

    Evaluate a polynomial in three variables with the given coefficients at the
    given points.

    The points can be given as numpy arrays as long as they can be broadcast
    against each other; all outputs will have this shape and type.

    The polynomial coefficients are given as a flat iterable (tuple, list, or
    array). They are ordered with the degree in `z` incrementing slowest,
    followed by those for `y` and `x`. For each degree k in `z`, it is assumed
    that all degrees in `x` and `y` are present up to a value of
    (i+j = ijmaxs[k]).

    Arguments:
        x, y, z (float or array): Value(s) of each variable at which to
            evaluate the polynomial.
        coefs (iterable of float): Coefficients of the polynomial.
        ijmaxs (iterable of int): Maximum degree in `x` and `y` for each degree
            of `z`.

    Returns:
        p (float or array): Value of the polynomial at the given point(s).
    """
    # Track whether the output should be scalar or array
    scalar = isscalar(x, y, z, coefs[0])
    out = np.broadcast(x, y, z, coefs[0])
    p, p2, p3 = [np.zeros(out.shape) for __ in range(3)]

    # Loop backwards through coefficients in z then y and x
    ind = -1
    for ijmax in ijmaxs[-1::-1]:
        p2[...] = 0.
        for j in range(ijmax, -1, -1):
            p3[...] = 0.
            for i in range(ijmax-j, -1, -1):
                p3 *= x
                p3 += coefs[ind]
                ind -= 1
            p2 *= y
            p2 += p3
        p *= z
        p += p2

    # Reformat for scalar output
    if scalar:
        p = float(p)
    return p


def poly3d_1der(x, y, z, coefs, ijmaxs, xscale=1., yscale=1., zscale=1.):
    """Evaluate a 3d polynomial with first derivatives.

    Evaluate a polynomial in three variables with the given coefficients at the
    given points, as well as the first derivatives with respect to each
    variable.

    The points can be given as numpy arrays as long as they can be broadcast
    against each other; all outputs will have this shape and type.

    The polynomial coefficients are given as a flat iterable (tuple, list, or
    array). They are ordered with the degree in `z` incrementing slowest,
    followed by those for `y` and `x`. For each degree k in `z`, it is assumed
    that all degrees in `x` and `y` are present up to a value of
    (i+j = ijmaxs[k]).

    Arguments:
        x, y, z (float or array): Value(s) of each variable at which to
            evaluate the polynomial.
        coefs (iterable of float): Coefficients of the polynomial.
        ijmaxs (iterable of int): Maximum degree in `x` and `y` for each degree
            of `z`.
        xscale, yscale, zscale (float, optional): Amount to scale each
            derivative by (default 1.). Useful for calculating physical
            derivatives.

    Returns:
        p (float or array): Value of the polynomial at the given point(s).
        px, py, pz (float or array): Derivative of the polynomial with respect
            to each variable.
    """
    # Track whether the output should be scalar or array
    scalar = isscalar(x, y, z, coefs[0])
    out = np.broadcast(x, y, z, coefs[0])
    p, px, py, pz, p2, py2, px2, p3, px3 = [
        np.zeros(out.shape) for __ in range(9)]

    # Loop backwards through coefficients in z then y
    ind = -1
    for ijmax in ijmaxs[-1::-1]:
        p2[...] = 0.
        py2[...] = 0.
        px2[...] = 0.
        for j in range(ijmax, -1, -1):
            p3[...] = 0.
            px3[...] = 0.
            for i in range(ijmax-j, -1, -1):
                px3 *= x
                px3 += p3
                p3 *= x
                p3 += coefs[ind]
                ind -= 1
            px2 *= y
            px2 += px3
            py2 *= y
            py2 += p2
            p2 *= y
            p2 += p3
        px *= z
        px += px2
        py *= z
        py += py2
        pz *= z
        pz += p
        p *= z
        p += p2

    # Reformat for scalar output
    px /= xscale
    py /= yscale
    pz /= zscale
    if scalar:
        ps = [p, px, py, pz]
        makescalar(ps)
        p, px, py, pz = ps[:]
    return (p, px, py, pz)


def poly3d_ders(x, y, z, coefs, ijmaxs, dmax, xscale=1., yscale=1., zscale=1.):
    """Evaluate a 3d polynomial with multiple derivatives.

    Evaluate a polynomial in three variables with the given coefficients at the
    given points, as well as the derivatives up to a specified maximum.

    The points can be given as numpy arrays as long as they can be broadcast
    against each other; all outputs will have this shape and type.

    The polynomial coefficients are given as a flat iterable (tuple, list, or
    array). They are ordered with the degree in `z` incrementing slowest,
    followed by those for `y` and `x`. For each degree k in `z`, it is assumed
    that all degrees in `x` and `y` are present up to a value of
    (i+j = ijmaxs[k]).

    Arguments:
        y, z (float or array): Value(s) of each variable at which to evaluate
            the polynomial.
        coefs (iterable of float): Coefficients of the polynomial.
        ijmaxs (iterable of int): Maximum degree in `x` and `y` for each degree
            of `z`.
        dmax (int >= 0): Maximum number of derivatives to take.
        xscale, yscale, zscale (float, optional): Amount to scale each
            derivative by (default 1.). Useful for calculating physical
            derivatives.

    Returns:
        ps (list of list of float or array): Value of the polynomial and its
            derivatives. The derivatives are returned with the x-derivative
            varying first, followed by the y- and z-derivatives. For dmax=2,
            the structure returned is
                ps = [[[p, px, pxx], [py, pxy], [pyy]],
                    [[pz, pxz], [pzy]], [[pzz]]].
    """
    # Track whether the output should be scalar or array
    scalar = isscalar(x, y, z, coefs[0])
    out = np.broadcast(x, y, z, coefs[0])
    ps = [[[np.zeros(out.shape) for dx in range(dmax+1-dz-dy)]
           for dy in range(dmax+1-dz)]for dz in range(dmax+1)]
    ps2 = [[np.zeros(out.shape) for dx in range(dmax+1-dy)]
           for dy in range(dmax+1)]
    ps3 = [np.zeros(out.shape) for dx in range(dmax+1)]

    # Loop backwards through coefficients in z then y
    ind = -1
    for ijmax in ijmaxs[-1::-1]:
        # Track values with only y-derivatives
        dy, dx = 0, 0
        for __ in range((dmax+1)*(dmax+2)//2):
            ps2[dy][dx][...] = 0.
            dxmax = dmax-dy
            dy = ((dy+1) if (dx == dxmax) else dy)
            dx = (0 if (dx == dxmax) else (dx+1))
        for j in range(ijmax, -1, -1):
            # Track values with only x-derivatives
            for dx in range(dmax+1):
                ps3[dx][...] = 0.
            i, dx = ijmax-j, dmax
            for __ in range((ijmax-j+1)*(dmax+1)):
                ps3[dx] *= x
                if dx == 0:
                    ps3[dx] += coefs[ind]
                    dx = dmax
                    ind -= 1
                    i -= 1
                else:
                    ps3[dx] += dx*ps3[dx-1]/xscale
                    dx -= 1

            # Calculate the y-derivative terms
            dx, dy = 0, dmax
            for __ in range((dmax+1)*(dmax+2)//2):
                ps2[dy][dx] *= y
                if dy == 0:
                    ps2[dy][dx] += ps3[dx]
                    dx += 1
                    dy = dmax-dx
                else:
                    ps2[dy][dx] += dy*ps2[dy-1][dx]/yscale
                    dy -= 1

        # Calculate the z-derivative terms
        dx, dy, dz = 0, 0, dmax
        for __ in range((dmax+1)*(dmax+2)*(dmax+3)//6):
            ps[dz][dy][dx] *= z
            ps[dz][dy][dx] += (
                ps2[dy][dx] if (dz == 0) else dz*ps[dz-1][dy][dx]/zscale)
            dymax = dmax-dx
            dx = ((dx+1) if (dy == dymax) else dx)
            dy = ((0 if (dy == dymax) else (dy+1)) if (dz == 0) else dy)
            dz = ((dmax-dx-dy) if (dz == 0) else dz-1)

    # Reformat for scalar output
    if scalar:
        makescalar(ps)
    return ps


# Root-finding methods
def update_newton(yvals, y0):
    """Calculate the variable increment using Newton's method.

    Calculate the amount to increment the variable by in one iteration of
    Newton's method. The goal is to find the value of x for which y(x) = y0.
    `yvals` contains the values [y(x0), y'(x0), ...] of the function and its
    derivatives at the current estimate of x. The Newton update increment is
        dx = (y0 - y)/y'.
    Newton's method iterates by adding x += dx at each step.

    Arguments:
        yvals (iterable of float or array): The value of y(x) and its
            derivatives at the current estimate of x. Must contain at least
            2 entries, for y and y'.
        y0 (float or array): The target value of y.

    Returns:
        dx (float or array): The update increment for x.
    """
    dx = (y0 - yvals[0])/yvals[1]
    return dx


def update_halley(yvals, y0):
    """Calculate the variable increment using Halley's method.

    Calculate the amount to increment the variable by in one iteration of
    Halley's method. The goal is to find the value of x for which y(x) = y0.
    `yvals` contains the values [y(x0), y'(x0), y''(x0), ...] of the function
    and its derivatives at the current estimate of x. The Halley update
    increment is
        dx0 = (y0 - y)/y'
        dx = dx0 / (1 + dx0*y''/(2*y')).
    Here, dx0 is the increment used by Newton's method. Halley's method
    iterates by adding x += dx at each step.

    Arguments:
        yvals (iterable of float or array): The value of y(x) and its
            derivatives at the current estimate of x. Must contain at least
            3 entries, for y, y' and y''.
        y0 (float or array): The target value of y.

    Returns:
        dx (float or array): The update increment for x.
    """
    dx0 = (y0 - yvals[0])/yvals[1]
    dx = dx0 / (1 + dx0*yvals[2]/(2*yvals[1]))
    return dx


# Define available update functions
UPDATERS = {'newton': update_newton, 'halley': update_halley}


def rootfinder(
        derfun, y0, x0, xmin, ymin, args, update='newton', maxiter=MAXITER,
        rxtol=None, axtol=None, rytol=RTOL, aytol=None):
    """Find the root of a function with an iterative method.

    Calculate the root of a function using a derivative-based iterative method,
    either Newton's method or Halley's. The goal is to calculate the value of x
    for which
        derfun(x, *args)[0] = y0
    where derfun returns a list with [y(x), y'(x), ...].

    Arguments:
        derfun (callable): The function to be calculated and its derivatives
            with respect to x:
                derfun(x, *args) = [y(x), y'(x), ...].
            It must return at least the first derivative to use Newton's method
            or the second derivative to use Halley's.
        y0 (float or array): Target value of y.
        x0 (float or array): Initial estimate of x.
        xmin (float): Minimum value for x when calculating relative step sizes.
        ymin (float): Minimum value for y when calculating relative errors.
        args (iterable of float or array, optional): Additional arguments to
            derfun. If None (default) no additional arguments are passed.
        update (str, optional): The name of the update method to use, either
            'newton' for Newton's method or 'halley' for Halley's method.
        maxiter (int, optional): Maximum number of iterations to allow before
            stopping (default `MAXITER`).
        rxtol (float, optional): Minimum change in x, relative to the current
            estimate, to allow in the iterations before stopping. If None
            (default) then the relative step size will not be checked.
        axtol (float, optional): Minimum change in x to allow before stopping.
            If None (default) then the absolute step size will not be checked.
        rytol (float, optional): Maximum error in y, relative to the target y0,
            to allow in the iterations before stopping (default `RTOL`). If
            None, then the relative error will not be checked.
        aytol (float, optional): Maximum error in y to allow in the iterations
            before stopping. If None (default) then the absolute error will not
            be checked.

    Returns:
        x (float or array): The best estimate of x.

    Raises:
        RuntimeWarning: If the maximum number of iterations is reached before
            the error gets below the given tolerances.
        RuntimeWarning: If the step size gets too small before the error gets
            below the given tolerances.
    """
    # Determine form of inputs
    updatefun = UPDATERS[update]
    arglist = list(args) + [x0, y0]
    scalar = isscalar(*arglist)
    if scalar:
        shape = None
        x = x0
    else:
        shape = np.broadcast(*arglist).shape
        x = np.zeros(shape)
        x[...] = x0
    rxtol, axtol = [(0. if (tol is None) else tol) for tol in (rxtol, axtol)]
    rytol, aytol = [(np.inf if (tol is None) else tol)
                    for tol in (rytol, aytol)]
    ryval = np.maximum(np.abs(y0), ymin)

    # Iteratively apply the root-finding update
    for it in range(maxiter):
        # Calculate y with current guess of x0
        yvals = derfun(x, *args)

        # Are the values sufficiently close?
        ayerr = np.abs(yvals[0] - y0)
        ryerr = ayerr/ryval
        if np.all(ayerr < aytol) and np.all(ryerr < rytol):
            break

        # Calculate increment in tpot
        dx = updatefun(yvals, y0)

        # Are the steps too small to continue?
        axstep = np.abs(dx)
        rxval = np.maximum(np.abs(x), xmin)
        rxstep = axstep/rxval
        if np.all(axstep < axtol) and np.all(rxstep < rxtol):
            msg = 'Step sizes are too small to continue.\n'
            msg += f'\tIteration: {it}\n'
            msg += f'\tMaximum absolute step: {np.max(axstep)}\n'
            msg += f'\tMaximum relative step: {np.max(rxstep)}\n'
            warnings.warn(msg, category=RuntimeWarning)
            break

        # Continue with the iteration
        x += dx
    else:
        # Maximum number of iterations reached
        msg = f'Maximum number {maxiter} of iterations reached.\n'
        msg += f'Maximum absolute error: {np.max(ayerr)}\n'
        msg += f'Maximum relative error: {np.max(ryerr)}\n'
        warnings.warn(msg, category=RuntimeWarning)

    # Reformat the final value if necessary
    if scalar:
        x = float(x)
    return x
