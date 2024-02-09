#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2023 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || l.brunner@univie.ac.at

Abstract: 

"""
import numpy as np
import xarray as xr
import pandas as pd
import regionmask


def calc_bias(da, percentile):
    if isinstance(da, float):
        pass
    elif 'time' in da.dims:
        da = (da > 0).mean('time')
    elif 'time2' in da.dims:
        da = (da > 0).mean('time2')
    elif 'dayofyear' in da.dims:
        da = da.mean('dayofyear')
        
    ref = 100 - percentile
    return (da * 100 - ref) / ref * 100


def spatial_aggregation(da, area_method='mean', ignore_inf=False):
    if ignore_inf:
        da = da.where(np.isfinite(da))
    if isinstance(area_method, int):
        # NOTE: using quantile here as weighted percentile is not implemented
        return da.weighted(np.cos(np.deg2rad(da['lat']))).quantile(area_method / 100., dim=('lat', 'lon')).squeeze()
    if area_method == 'mean':
        return da.weighted(np.cos(np.deg2rad(da['lat']))).mean(('lat', 'lon')).squeeze()

    raise ValueError('Unknown area_method: {}'.format(area_method))
    
    
def mask_ocean(da):
    mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(da) == 0
    return da.where(mask)


def delete_short_periods(da, min_duration=3):
    """Set blocks of True to False if their length is < min_duration.
    
    We do this here by utilizing a gap interpolation routine:
    - set all true values to nan
    - interpolate False over short nan periods
    - set remaining nans back to True
    """
    timen = 'time'
    if 'time2' in da.dims:
        timen = 'time2'
    max_gap = pd.Timedelta(min_duration + .5, 'day')

    da_long = da.fillna(False).astype(bool)  # set any existing nans to False
    da_long = da_long.where(~da_long)  # set extremes (true) days to nan
    da_long = da_long.interpolate_na(timen, max_gap=max_gap)  # interpolate zeros over nan periods shorter than min_duration days
    da_long = da_long.fillna(True).astype(bool)  # set remaining nan back to true
    da_long = da_long.where(np.isfinite(da))  # overwrite False with original nans
    return da_long


# # TODO: replace?
# def delete_short_episodes(data, min_length=3):
#     data = np.copy(data)
#     idx_start, episode_length = find_episodes(data > 0)
#     idx_del = np.where(episode_length < min_length)[0]
#     for idx in idx_del:
#         data[idx_start[idx]:idx_start[idx] + episode_length[idx]] = 0
        
#     return data



# def find_episodes(arr):
#     """Find runs of consecutive items in an array."""

#     # ensure array
#     arr = np.asanyarray(arr)
#     if arr.ndim != 1:
#         raise ValueError('only 1D array supported')
#     n = arr.shape[0]

#     # handle empty array
#     if n == 0:
#         return np.array([]), np.array([]), np.array([])

#     # find run starts
#     loc_run_start = np.empty(n, dtype=bool)
#     loc_run_start[0] = True
#     np.not_equal(arr[:-1], arr[1:], out=loc_run_start[1:])
#     run_starts = np.nonzero(loc_run_start)[0]

#     # find run values
#     run_values = arr[loc_run_start]

#     # find run lengths
#     run_lengths = np.diff(np.append(run_starts, n))

#     return run_starts[run_values], run_lengths[run_values]
