#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2023 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || l.brunner@univie.ac.at

Abstract: Calculate thresholds and thrshold exceedandes for different cases.
"""

import argparse
import os
import numpy as np
import xarray as xr
import logging

from logger_functions import set_logger, LogTime
from io_functions import get_filename

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 365


def parse_args():
    """Parse command line input."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        dest="input",
        type=str,
        help="Input file path (needs to contain a 3D array)",
    )

    parser.add_argument(
        "--window",
        "-w",
        dest="window",
        default=31,
        type=int,
        help="Window size in days, has to be odd. Optional, default is 31.",
    )

    parser.add_argument(
        "--percentile",
        "-p",
        dest="percentile",
        default=90,
        type=int,
        help="Percentile to calculate. Optional, default is 90.",
    )

    parser.add_argument(
        '--startyear-base',
        dest='startyear_base',
        default='1961',
        type=str,
        help='Start year for the base period (default: 1961).'
    )

    parser.add_argument(
        '--endyear-base',
        dest='endyear_base',
        default='1990',
        type=str,
        help='End year for the base period (default: 1990).'
    )
    
    parser.add_argument(
        '--startyear-test',
        dest='startyear_test',
        default=None,
        type=str,
        help='Start year for the test period. If None same as base period.'
    )

    parser.add_argument(
        '--endyear-test',
        dest='endyear_test',
        default=None,
        type=str,
        help='End year for the test period. If None same as base period.'
    )

    args = parser.parse_args()
    
    if args.startyear_test is None and args.endyear_test is None:
        args.startyear_test = args.startyear_base
        args.endyear_test = args.endyear_base
    
    return args

    
def test_periods(da_base, da_test):
    """Make sure the periods have exactly DAYS_PER_YEAR days per year in every year."""
    years_base = np.unique(da_base['time.year'].values)
    years_test = np.unique(da_test['time.year'].values)
    
    if len(years_test) != len(years_base):
        print(years_test)
        print(years_base)
        raise ValueError("Base and test period have to have the same length.")
    
    doys_base, counts_base = np.unique(da_base['time.dayofyear'].values, return_counts=True)
    doys_test, counts_test = np.unique(da_test['time.dayofyear'].values, return_counts=True)
    
    if len(doys_base) != DAYS_PER_YEAR:
        raise ValueError("Base period has to cover all days of the year.")
    if len(doys_test) != DAYS_PER_YEAR:
        raise ValueError("Test period has to cover all days of the year.")
    
    if len(np.unique(counts_base)) != 1:
        raise ValueError("Base period has to have constant days per year.")
    if len(np.unique(counts_test)) != 1:
        raise ValueError("Test period has to have constant days per year.")
    
    if counts_base[0] != len(years_base):
        raise ValueError("Each day needs to exist exactly number of years times in base period.")
    if counts_test[0] != len(years_test):
        raise ValueError("Each day needs to exist exactly number of years times in test period.")
    
    return len(years_base)
    

def get_window_indices(pm_days: int) -> np.ndarray: 
    """Get indices for a single year and each day of the year.
    
    Parameters
    ----------
    pm_days : integer
        Window size as plus/minus the days around the central day. 
        This enforces windows to be odd, pm_days=0 is equivalent to 
        no running window.
        
    Returns
    -------
    np.ndarray : shape (DAYS_PER_YEAR, 2*pm_days + 1)
        Window indices for each day of the year.
    """
    idxs = np.arange(0, DAYS_PER_YEAR)
    idxs_2d = idxs.reshape(-1, 1) + np.arange(-pm_days, pm_days + 1)
    idxs_2d[idxs_2d < 0] += DAYS_PER_YEAR
    idxs_2d[idxs_2d >= DAYS_PER_YEAR] -= DAYS_PER_YEAR
  
    return idxs_2d
    
    
def get_window_indices_years(pm_days: int, years: int) -> np.ndarray:
    """Get indices for multiple years and each day of the year.
    
    Parameters
    ----------
    pm_days : integer
        Window size as plus/minus the days around the central day. 
        This enforces windows to be odd, pm_days=0 is equivalent to 
        no running window.
    years : integer
        Number of years.
        
    Returns
    -------
    np.ndarray : shape (DAYS_PER_YEAR, (2*pm_days + 1) * years)
        Window indices for each day of the year across multiple years.
    """
    idxs_2d = get_window_indices(pm_days)
    years_start = np.arange(0, years*DAYS_PER_YEAR, DAYS_PER_YEAR)
    idxs_3d = years_start.reshape(-1, 1, 1) + idxs_2d
    
    idxs_test = (years_start.reshape(-1, 1) + np.arange(0, DAYS_PER_YEAR)).swapaxes(0, 1)
    return idxs_3d.swapaxes(0, 1).reshape(DAYS_PER_YEAR, -1), idxs_test


def calc_threshold(arr: np.ndarray, idx_thr: np.ndarray, percentile: int) -> np.ndarray:
    """Calculate the given percentile for each day of the year using the window given by idx_thr."""
    if isinstance(percentile, str) and percentile == 'mean':
        return np.mean(arr[idx_thr], axis=-1)
    return np.percentile(arr[idx_thr], percentile, axis=-1)


def calc_exceedances(arr: np.ndarray, thr: np.ndarray, idx_test: np.ndarray) -> np.ndarray:
    """Calculate the difference to the threshold for each day of the year."""
    return (arr[idx_test] - thr.reshape(-1, 1)).ravel('F')


def main(input_, window, percentile, startyear_base, endyear_base, startyear_test, endyear_test):
    log = LogTime()
    
    if window % 2 != 1:
        raise ValueError('Window needs to be odd not {}'.format(window))
    pm_days = int((window - 1) / 2) 
    
    log.start('Build output file name and check if it already exists')
    fn_output = get_filename(
        dataset='era5' if 'era5' in input_ else os.path.basename(input_).split('_')[2],
        startyear_base=startyear_base, 
        endyear_base=endyear_base,
        window=window,
        percentile= percentile,
        startyear_test=startyear_test,
        endyear_test=endyear_test
    )

    if os.path.exists(fn_output):
        logger.warning("Output file {} already exists. Skipping.".format(fn_output))
        return None
    logger.info("Output file: {}".format(fn_output))
        
    log.start('Opening input file')
    if 'era5' in input_:        
        da = xr.open_dataset(input_, use_cftime=True)['t2m']  
    else:  # CMIP6
        da_hist = xr.open_dataset(input_, use_cftime=True)['tasmax']
        da_fut = xr.open_dataset(input_.replace('historical', 'ssp370'), use_cftime=True)['tasmax']
        da = xr.concat([da_hist, da_fut], dim='time')
        input_ = ', '.join([input_, input_.replace('historical', 'ssp370')])
    
    log.start('Converting to noleap calendar')    
    da = da.convert_calendar('noleap')
    
    da_base = da.sel(time=slice(startyear_base, endyear_base))
    da_test = da.sel(time=slice(startyear_test, endyear_test))
    nr_years = test_periods(da_base, da_test)
    
    log.start('Calculate and remove seasonal cycle')
    idx_thr, idx_test = get_window_indices_years(0, nr_years)

    # NOTE: Reuse the same functions for calculating the mean seasonal cycle
    # (therefore the function names are a bit misleading)
    # This gives us the option to use a running window for the mean seasonal cycle
    # (not done at the momet)
    mean_seas_cycle = xr.apply_ufunc(
        calc_threshold,
        da_base, 
        input_core_dims=[['time']],
        output_core_dims=[['dayofyear']],
        kwargs={'idx_thr': idx_thr, 'percentile': 'mean'},
        vectorize=True)
    da_base_des = xr.apply_ufunc(
        calc_exceedances,
        da_base, mean_seas_cycle,
        input_core_dims=[['time'], ['dayofyear']],
        output_core_dims=[['time']],
        kwargs={'idx_test': idx_test},
        vectorize=True)
    da_test_des = xr.apply_ufunc(
        calc_exceedances,
        da_test, mean_seas_cycle,
        input_core_dims=[['time'], ['dayofyear']],
        output_core_dims=[['time']],
        kwargs={'idx_test': idx_test},
        vectorize=True)
        
    log.start('Calculating percentiles with seasonal cycle')
    idx_thr, idx_test = get_window_indices_years(pm_days, nr_years)
    threshold = xr.apply_ufunc(
        calc_threshold,
        da_base, 
        input_core_dims=[['time']],
        output_core_dims=[['dayofyear']],
        kwargs={'idx_thr': idx_thr, 'percentile': percentile},
        vectorize=True)
    exceedances = xr.apply_ufunc(
        calc_exceedances,
        da_base, threshold,
        input_core_dims=[['time'], ['dayofyear']],
        output_core_dims=[['time']],
        kwargs={'idx_test': idx_test},
        vectorize=True)
    
    log.start('Calculating percentiles without seasonal cycle')
    threshold_des = xr.apply_ufunc(
        calc_threshold,
        da_base_des, 
        input_core_dims=[['time']],
        output_core_dims=[['dayofyear']],
        kwargs={'idx_thr': idx_thr, 'percentile': percentile},
        vectorize=True)
    exceedances_des = xr.apply_ufunc(
        calc_exceedances,
        da_base_des, threshold_des,
        input_core_dims=[['time'], ['dayofyear']],
        output_core_dims=[['time']],
        kwargs={'idx_test': idx_test},
        vectorize=True)
    
    log.start('Merge DataArrays to Dataset')
    
    ds = xr.Dataset(
        {
            'tasmax': da_base,
            'tasmax_deseasonalized': da_base_des,
            'seasonal_cycle': mean_seas_cycle,
            'threshold': threshold, 
            'threshold_deseasonalized': threshold_des,
            'exceedances': exceedances, 
            'exceedances_deseasonalized': exceedances_des,
            },
    )  
    ds.attrs['window'] = window
    ds.attrs['percentile'] = percentile
    ds.attrs['source_file'] = input_
    
    if startyear_base != startyear_test:
        ds['tasmax_test'] = da_test.rename({'time': 'time2'})  # seconde time dimension
        ds['tasmax_test_deseasonalized'] = da_test_des.rename({'time': 'time2'})

        log.start('Calculating percentiles test period')
        exceedances_test = xr.apply_ufunc(
            calc_exceedances,
            da_test, threshold,
            input_core_dims=[['time'], ['dayofyear']],
            output_core_dims=[['time']],
            kwargs={'idx_test': idx_test},
            vectorize=True)
        exceedances_test_des = xr.apply_ufunc(
            calc_exceedances,
            da_test_des, threshold_des,
            input_core_dims=[['time'], ['dayofyear']],
            output_core_dims=[['time']],
            kwargs={'idx_test': idx_test},
            vectorize=True)
        ds['exceedances_test'] = exceedances_test.rename({'time': 'time2'})  # seconde time dimension
        ds['exceedances_test_deseasonalized'] = exceedances_test_des.rename({'time': 'time2'})
    
    log.start('Save file')
    encoding = {varn: {'dtype' : 'float32'} for varn in ds if 'time' not in varn}
    ds.to_netcdf(fn_output, encoding=encoding)
    log.stop
    
    return ds


if __name__ == '__main__':
    args = parse_args()
    set_logger()

    for arg, value in sorted(vars(args).items()):
        logger.info("{}: {}".format(arg, value))

    with LogTime(os.path.basename(__file__).replace('py', 'main()')):
        main(
            input_=args.input,
            window=args.window,
            percentile=args.percentile,
            startyear_base=args.startyear_base,
            endyear_base=args.endyear_base,
            startyear_test=args.startyear_test,
            endyear_test=args.endyear_test,
        )
