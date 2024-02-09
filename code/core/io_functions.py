#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2023 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || l.brunner@univie.ac.at

Abstract: Input-output functions. Change basepath here for different file systems.

"""
import os
import xarray as xr
import numpy as np

startyear = 1961
endyear = 1990

PLOT_PATH = '../figures'
DATA_PATH = '../data'

model_names = [
    'ACCESS-CM2',
    'ACCESS-ESM1-5',
    'AWI-CM-1-1-MR',
    'BCC-CSM2-MR',
    'CanESM5',
    'CMCC-ESM2',
    'EC-Earth3-AerChem',
    'EC-Earth3',
    'EC-Earth3-Veg-LR',
    'EC-Earth3-Veg',
    'FGOALS-g3',
    'GFDL-ESM4',
    'INM-CM4-8',
    'INM-CM5-0',
    'IPSL-CM6A-LR',
    'MIROC6',
    'MPI-ESM1-2-HR',
    'MPI-ESM1-2-LR',
    'MRI-ESM2-0',
    'NorESM2-LM',
    'NorESM2-MM',
    'TaiESM1',
    # 'CAMS-CSM1-0',  # excluded in revision 1 due to missing year 2100
    'CNRM-CM6-1',
    'MIROC-ES2L',
    'GISS-E2-1-G',
    'CNRM-ESM2-1',
]


def get_filename(
    dataset: str='era5', 
    startyear_base: int=1961, 
    endyear_base: int=1990,
    window: int=31,
    percentile: int=90,
    startyear_test=None,
    endyear_test=None,
) -> str:
    """Get the filename for specific case."""
    if startyear_test is None and endyear_test is None:
        startyear_test = startyear_base
        endyear_test = endyear_base
    if startyear_test is None or endyear_test is None:
        raise ValueError(' '.join([
            'startyear_test and endyear_test need to be either both',
            'be set or both be None not {}, {}'.format(
                startyear_test, endyear_test)]))
        
    fn = os.path.join(
        DATA_PATH, 'frequencies',
        '_'.join([
            dataset,
            'b{}-{}'.format(startyear_base, endyear_base),
            't{}-{}'.format(startyear_test, endyear_test),
            'w{}'.format(window),
            'p{}'.format(percentile)
        ])) + '.nc'

    return fn