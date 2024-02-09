#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2023 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || l.brunner@univie.ac.at

Abstract: Functions to create synthetic time series from white noise.
"""
import numpy as np

DAYS_PER_YEAR = 365
YEARS = 30
PERCENTILE = 90
RANDOM_SEED = 0


# https://stackoverflow.com/questions/33898665/python-generate-array-of-specific-autocorrelation
def white_noise_autocorrelated(n_samples: int, corr: float, random_seed: int=RANDOM_SEED, mu: float=0, sigma: float=1) -> np.ndarray:
    """Create a white noise time series with a given auto-correlation.

    Parameters
    ----------
    n_samples : int
        The number of samples in the time series.
    corr : float
        The desired auto-correlation. Must be between 0 and 1.
    random_seed : int, optional by default RANDOM_SEED
    mu : float, optional by default 0
        The mean of the white noise.
    sigma : float, optional by default 1
        The standard deviation of the white noise.

    Returns
    -------
    np.ndarray, shape (n_samples,)
    """
    assert 0 <= corr < 1, "Auto-correlation must be between 0 and 1"

    # Find out the offset `c` and the std of the white noise `sigma_e`
    # that produce a signal with the desired mean and variance.
    # See https://en.wikipedia.org/wiki/Autoregressive_model
    # under section "Example: An AR(1) process".
    c = mu * (1 - corr)
    sigma_e = np.sqrt((sigma ** 2) * (1 - corr ** 2))

    # Sample the auto-regressive process.
    signal = [c + np.random.RandomState(random_seed).normal(0, sigma_e)]
    for idx in range(1, n_samples):
        signal.append(c + corr * signal[-1] + np.random.RandomState(random_seed + idx + 34578).normal(0, sigma_e))

    return np.array(signal)


def seasonal_cycle_sin(years: int=YEARS, days_per_year: int=DAYS_PER_YEAR, amplitude: float=3) -> np.ndarray:
    """Create a seasonal cycle time series based on a sine function.

    Parameters
    ----------
    years : int, optional by default YEARS
        The number of years in the time series.
    days_per_year : int, optional by default DAYS_PER_YEAR
        The number of days per year in the time series.
    amplitude : int, optional by default 3
        The amplitude of the sine function.

    Returns
    -------
    np.ndarray, shape (years*days_per_year,)
    """
    xx = np.arange(years*days_per_year) / days_per_year * 2 * np.pi
    return np.sin(xx) / np.std(np.sin(xx[:days_per_year])) * amplitude 
    
    
def synthetic_timeseries(
        sc_amplitude: float=3, noise_amplitude: float=1, auto_corr: float=.8, 
        random_seed: int=RANDOM_SEED, years: int=YEARS, days_per_year: int=DAYS_PER_YEAR) -> np.ndarray:
    """Create a synthetic time series with a seasonal cycle and auto-correlated noise.

    Parameters
    ----------
    sc_amplitude : float, optional by default 3
        The amplitude of the seasonal cycle.
    noise_amplitude : float, optional by default 1
        The amplitude of the noise.
    auto_corr : float, optional by default .8
        The auto-correlation of the noise.
    random_seed : int, optional by default RANDOM_SEED
    years : int, optional by default YEARS
        The number of years in the time series.
    days_per_year : int, optional by default DAYS_PER_YEAR
        The number of days per year in the time series.
    **kwargs : dict, optional by default {}
        Disregarded.
    """
    ts = white_noise_autocorrelated(years*days_per_year, corr=auto_corr, random_seed=random_seed, sigma=noise_amplitude)
    return ts + seasonal_cycle_sin(years, days_per_year, amplitude=sc_amplitude)


def get_window_indices(pm_days: int, years: int=YEARS, days_per_year: int=DAYS_PER_YEAR) -> np.ndarray:
    """Return a nested array giving running window indices for each day of the year.
    
    Parameters
    ----------
    pm_days : int
        Half the size of the running window -1. I.e, plus/minus the number of days on 
        each side of the central day. This enforces the total window to be odd.
    years : int, optional, by default YEARS
    days_per_year : int, optional, by default DAYS_PER_YEAR
    
    Returns
    -------
    windows : np.ndarray, shape (days_per_year, years*(2*pm_days+1))
    """
    idx_doy = np.arange(days_per_year)
    idx_win = np.arange(-pm_days, pm_days+1, 1)
    days_total = years*days_per_year
    
    # build window indices for the first year
    idx_win_y = np.add(idx_win.reshape(1, -1), idx_doy.reshape(-1, 1))
    
    # build window indices for all years
    idx_win_all = np.add(
        idx_win_y.reshape(*idx_win_y.shape, 1),
        np.arange(0, days_total, days_per_year).reshape(1, -1)).reshape(days_per_year, -1)
    
    # fix beginning and end
    idx_win_all[idx_win_all >= days_total] -= days_total
    idx_win_all[idx_win_all < 0] += days_total
    
    return idx_win_all


def calc_threshold(data, window_indices=None, percentile=PERCENTILE, pm_days=None, days_per_year=DAYS_PER_YEAR, **kwargs) -> np.ndarray:
    """Calculate the threshold for each day of the year for a given percentile and window.
    
    Parameters
    ----------
    data : np.ndarray, shape (years*days_per_year)
    percentile : float in (0, 1)
    window_indices : np.ndarray, optional, by default None
        Output of get_window_indices. Can be None, then years and pm_days need to be 
        given and it will be calculated on the fly.
    years : int, optional, by default None
        Only valid if window_indices is None. Number of years in data.
    pm_days : int, optional, by default None
        Only valid if window_indices is None. Half the size of the running window -1
    kwargs : dict, optional, by default {}
        Passed on to np.percentile. Can be used to change the percentile method.
        
    Returns
    -------
    threshold : np.ndarray, shape (days_in_year)
    """
    if window_indices is None:
        if len(data) % days_per_year != 0:
            raise ValueError
        window_indices = get_window_indices(pm_days, years=len(data) // days_per_year, days_per_year=days_per_year)
    return np.percentile(data[window_indices], percentile, axis=1, **kwargs)


def calc_exceedances(data: np.ndarray, threshold: np.ndarray, frequency=True) -> np.ndarray:
    if len(data) % DAYS_PER_YEAR != 0:
        raise ValueError
    
    if len(threshold.shape) == 1:
        threshold = threshold.reshape(1, -1)
    threshold = threshold.swapaxes(0, 1)
    exceedances = data.reshape(-1, DAYS_PER_YEAR, 1) - threshold.reshape(1, *threshold.shape)
    
    if frequency:
        return (exceedances > 0).mean(axis=0)
    return exceedances.reshape(-1, threshold.shape[1])


def get_data_threshold_exceedances(
    pm_days, random_seed=RANDOM_SEED, in_sample=True,
    years=YEARS, days_per_year=DAYS_PER_YEAR,
    sc_amplitude=3, noise_amplitude=1, auto_corr=.8,
    percentile=PERCENTILE, return_frequencies=False, **kwargs):
    """Wrapper for synthetic_timeseries, calc_threshold, and calc_exceedances."""
    
    data = synthetic_timeseries(
        sc_amplitude=sc_amplitude, noise_amplitude=noise_amplitude, auto_corr=auto_corr, 
        years=years, days_per_year=days_per_year, random_seed=random_seed)    
    threshold = calc_threshold(
        data, percentile=percentile, pm_days=pm_days, 
        days_per_year=days_per_year, **kwargs)
    if not in_sample:
        data = synthetic_timeseries(
            sc_amplitude=sc_amplitude, noise_amplitude=noise_amplitude, auto_corr=auto_corr, 
        years=years, days_per_year=days_per_year, random_seed=random_seed + 191681)   
    exceedances = calc_exceedances(data, threshold, frequency=return_frequencies)
    return data, threshold, exceedances


def get_bootstrap_exceedances(
    nr_samples: int, pm_days: int, in_sample: bool=True, 
    random_seed: int=RANDOM_SEED, 
    years=YEARS, days_per_year=DAYS_PER_YEAR,
    sc_amplitude=3, noise_amplitude=1, auto_corr=.8,
    percentile=PERCENTILE, return_frequencies=True, **kwargs) -> np.ndarray:
    """Wrapper for multiple cales of synthetic_timeseries, calc_threshold, and calc_exceedances."""
    exceedances_bootstrap = []
    window_indices = get_window_indices(pm_days, years, days_per_year)
    for idx in range(nr_samples):
        data = synthetic_timeseries(
            sc_amplitude=sc_amplitude, noise_amplitude=noise_amplitude, auto_corr=auto_corr,
            random_seed=random_seed + idx*435, years=years, days_per_year=days_per_year)
        threshold = calc_threshold(data, window_indices, percentile=percentile, **kwargs)
        if not in_sample:
            data = synthetic_timeseries(
                sc_amplitude=sc_amplitude, noise_amplitude=noise_amplitude, auto_corr=auto_corr,
                random_seed=random_seed + 191681 + idx*435, years=years, days_per_year=days_per_year)
        exceedances_bootstrap.append(calc_exceedances(data, threshold, frequency=return_frequencies))
    return np.array(exceedances_bootstrap)


def calc_frequencies(exceedances, days_per_year):
    if len(exceedances.shape) == 1:
        return (exceedances > 0).reshape(-1, days_per_year).mean(axis=0)
    return (exceedances > 0).reshape(len(exceedances), -1, days_per_year).mean(axis=1)
    
    
def find_episodes(arr):
    """Find runs of consecutive items in an array."""

    # ensure array
    arr = np.asanyarray(arr)
    if arr.ndim != 1:
        raise ValueError('only 1D array supported')
    n = arr.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    # find run starts
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(arr[:-1], arr[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]

    # find run values
    run_values = arr[loc_run_start]

    # find run lengths
    run_lengths = np.diff(np.append(run_starts, n))

    return run_starts[run_values], run_lengths[run_values]


def delete_short_episodes(data, min_length=3):
    data = np.copy(data)
    idx_start, episode_length = find_episodes(data > 0)
    idx_del = np.where(episode_length < min_length)[0]
    for idx in idx_del:
        data[idx_start[idx]:idx_start[idx] + episode_length[idx]] = 0
        
    return data