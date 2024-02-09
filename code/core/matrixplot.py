#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2023 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || l.brunner@univie.ac.at

Abstract: Functions to create a matrix of spatially aggregated results.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon, Patch

from core.io_functions import get_filename
from core.core_functions import spatial_aggregation, calc_bias

savepath = '/mnt/users/staff/lbrunner/Files/Projects/seasonal_extremes/'


def load_aggregate_data(window, percentile, deseasonalized=False):
    fn = get_filename(window=window, percentile=percentile)
    varn = 'exceedances_deseasonalized' if deseasonalized else 'exceedances'
    ref = 100 - percentile
    
    freq = (xr.open_dataset(fn)[varn] > 0).mean('time')
    ds = xr.Dataset(
        {'frequency': (('window', 'percentile', 'aggregation'), [[[
            spatial_aggregation(calc_bias(freq, percentile), 5), 
            spatial_aggregation(calc_bias(freq, percentile)),  # mean
            spatial_aggregation(calc_bias(freq, percentile), 95),
        ]]]),
         'aggregation': ('aggregation', [5, 50, 95]),
         'window': [window],
         'percentile': [percentile],
        })
    # The spatial inhomogeneity is defined as the difference between high and low bias regions
    ds['inhomogeneity'] = ds['frequency'].sel(aggregation=95) - ds['frequency'].sel(aggregation=5)
    return ds


def load_data_matrix(windows=[1, 5, 15, 31, 45], percentiles=[50, 90, 95, 98, 99], deseasonalized=False):
    ds_list = []
    for window in windows:
        for percentile in percentiles:
            ds = load_aggregate_data(window, percentile, deseasonalized=deseasonalized)
            ds_list.append(ds)
    return xr.merge(ds_list)


def matrixplot(ds, vmax=None):
    # avoid the darkest colors to that the text staty well-readable
    cmap = plt.cm.get_cmap('RdBu_r', 265)
    colors = list(cmap(np.arange(265)))[132:-20]
    cmap = matplotlib.colors.ListedColormap(colors, "")
    cmap.set_bad('lightgray')
    
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(left=.1, right=.97, bottom=.12, top=.9)

    # use this case for labeling (not interesing)
    ds['inhomogeneity'].values[0, 0] = np.nan
    im = ax.imshow(np.around(ds['inhomogeneity'], 2), cmap=cmap, aspect='auto', vmax=vmax)
    
    # get rid of the negative sign for rounded positive biases
    ds['frequency'].values = np.around(ds['frequency'].values, 1)
    ds['frequency'].values[ds['frequency'] >= 0] = np.abs(ds['frequency'].values[ds['frequency'] >= 0])
    
    poly = Polygon(
        [[.5, 2.5], [.5, 3.5], [1.5, 3.5], [1.5, 2.5]],
        facecolor='none',
        edgecolor='k',
        lw=2,
        closed=True)
    ax.add_patch(poly)

    ax.set_yticks(np.arange(ds['window'].size), labels=ds['window'].values)
    ax.set_ylabel('Running window size (day)')

    ax.set_xticks(np.arange(ds['percentile'].size), labels=ds['percentile'].values)
    ax.set_xlabel('Percentile', labelpad=1)

    for xx, percentile in enumerate(ds['percentile']):
        for yy, window in enumerate(ds['window']):

            if xx == 0 and yy == 0:
                ax.text(xx, yy - .35, 'Inhomogeneity', 
                        ha='center', va='top', color='k', fontweight='bold', fontsize='small')
                ax.text(
                    xx, yy - .05, '5th/95th perc\nMean',
                    ha="center", va="top", color='k', fontsize='small')
                
            else:
                ax.text(xx, yy - .35, '{:.1f}%'.format(
                    np.around(ds['inhomogeneity'].sel(percentile=percentile, window=window), 1)), 
                        ha='center', va='top', color='k', fontweight='bold', fontsize='small')
                
                ax.text(
                    xx, yy - .05, '{:.1f}%/{:.1f}%\n{:.1f}%'.format(
                        ds['frequency'].sel(aggregation=5, percentile=percentile, window=window),
                        ds['frequency'].sel(aggregation=95, percentile=percentile, window=window),
                        ds['frequency'].sel(aggregation=50, percentile=percentile, window=window),
                        ),
                    ha="center", va="top", color='k', fontsize='small')
                
    return fig, ax