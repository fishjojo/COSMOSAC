#!/usr/bin/env python
import cCOSMO
import numpy as np
import os
import timeit

here = os.path.abspath(os.path.dirname(__file__))

def calc_LNAC(*, method, dbname, filenames, names, composition, T):

    if dbname == 'UD':
        db = cCOSMO.EmptyDatabase()
    else:
        raise RuntimeError

    # Add the fluids we want into the database
    for filename, name in zip(filenames, names):
        db.add_profile(filename, name)

    assert(len(names) == len(composition))
    if method == 'COSMOSAC-2002':
        COSMO = cCOSMO.COSMO1(names, db)
    elif method in ['COSMOSAC-2010','COSMOSAC-dsp']:
        COSMO = cCOSMO.COSMO3(names, db)

        # Specialize for 2010, no dispersive contribution, but residual and combinatorial
        if method  == 'COSMOSAC-2010':
            return COSMO.get_lngamma_comb(T, composition) + COSMO.get_lngamma_resid(T, composition)
    else:
        raise ValueError('Invalid method: ' + method)

    # If we haven't already returned, then do the calculation of ln(gamma)
    return COSMO.get_lngamma(T, composition)

if __name__ == '__main__':

    method = 'COSMOSAC-dsp' #'COSMOSAC-2002','COSMOSAC-2010','COSMOSAC-dsp'
    T = 300
    xi = 0.3
    composition = [xi, 1-xi]
    filenames = ['ethanol.sigma.1', 'h2o.sigma']
    names = ['ethanol', 'water']
    try:
        print(method, calc_LNAC(T=T, dbname='UD', filenames=filenames, names=names, composition = composition, method=method))
    except BaseException as BE:
        print(BE)
