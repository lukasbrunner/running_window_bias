#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2023 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || l.brunner@univie.ac.at

Abstract: 

"""
import os
import numpy as np
import xarray as xr
import logging

from io_functions import DATA_PATH
from synthetic_timeseries_functions import get_bootstrap_exceedances
from logger_functions import set_logger

logger = logging.getLogger(__name__)
set_logger()


# for figure 2
samples = 5000
pm_days_list = np.array([2, 15])
percentiles = np.arange(80, 100, 1)
sc_amplitudes = np.array([0, 1.8, 3])

for pm_days in pm_days_list:
    da_list = []
    logger.info('pm_days: {}'.format(pm_days))
    for sc_amplitude in sc_amplitudes:
        fn = os.path.join(DATA_PATH, 'synthetic', 'synthetic_percentiles_bootstrap_sc{}_w{}_test_weibull.nc'.format(sc_amplitude, 2*pm_days + 1))
        if os.path.isfile(fn):
            continue
        logger.info('sc_amplitude: {}'.format(sc_amplitude))
        frequency = get_bootstrap_exceedances(
            samples, pm_days, 
            percentile=percentiles,
            sc_amplitude=sc_amplitude,
            in_sample=False,  # in-sample/out-of-sample
            method='weibull',  # percentile method
        ).mean(axis=(0, 1))

        da = xr.DataArray(
            data=frequency.reshape(-1, 1, 1), 
            coords={
                'percentile': percentiles,
                'seasonal_cycle_amplitude': [sc_amplitude],
                'window': [2*pm_days + 1]
            },
            name='frequency_bootstrap',
        )
        ds = da.to_dataset()
        ds.to_netcdf(fn)
        logger.info('sc_amplitude: {} DONE'.format(sc_amplitude))
    
