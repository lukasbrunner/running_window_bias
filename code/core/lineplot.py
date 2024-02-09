#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2023 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || l.brunner@univie.ac.at

Abstract: Functions to create lineplots of the seasonal cycle.
"""
import numpy as np
import matplotlib.pyplot as plt

from core.core_functions import calc_bias

# some nice colors
cmap = plt.cm.get_cmap('RdBu_r', 20)
red = list(cmap(np.arange(20)))[-3]
blue = list(cmap(np.arange(20)))[3]

YEARS = 30


def set_xticks(ax, german=False):
    ax.set_xticks([0, 59, 120, 181, 243, 304, 364])
    ticklabels = ax.set_xticklabels(['1. Jan', '1. Mar', '1. May', '1. Jul', '1. Sep', '1. Nov', '31. Dec'])
    if german:
        ticklabels = ax.set_xticklabels(['1. Jan', '1. Mär', '1. Mai', '1. Jul', '1. Sep', '1. Nov', '31. Dez'])
        
    # could align first & last tick differently to save space but it does not look nice
    # ticklabels[0].set_ha("left")
    # ticklabels[-1].set_ha("right")
    ax.set_xlim(0, 364)

    
def lineplot(ax, data, years=YEARS, threshold=None, show_exceedances=True, show_years=True, show_legend=True, ylim=None, legend_loc=None, color=red, german=False):
    if len(data) % years != 0:
        raise ValueError    
    days_per_year = len(data) // years
    if days_per_year != 365:
        raise ValueError
    
    if show_years:
        label = '{} years'.format(years)
        if german:
            label = '{} Jahre'.format(years)
        ax.plot([], color='k', lw=.5, alpha=.5, label=label) # for legend
        
    for data_year in data.reshape(years, -1):
        if show_years:
            ax.plot(np.arange(days_per_year), data_year, color='k', lw=.1, alpha=.1)
        if threshold is not None and show_exceedances:
            ax.fill_between(
                np.arange(days_per_year), threshold, data_year, 
                where=data_year > threshold,
                facecolor=color, edgecolor='none', alpha=.5,
                interpolate=True,
            )
            
    if threshold is not None:
        label = 'Threshold'
        if german:
            label = 'Grenzwert'
        ax.plot(np.arange(days_per_year), threshold, color=color, lw=2, label=label)

    if threshold is not None and show_exceedances:
        label = 'Extremes'
        if german:
            label = 'Extreme'
        ax.fill_between(
            [], [], [],
            facecolor=color, edgecolor='none', alpha=.5,
            label=label
            )
        
    set_xticks(ax, german)
    if data.mean() > 200:  # assume absolute temperatures
        ax.set_ylabel('Temperature (K)')
    else:
        ax.set_ylabel('Temperature anomaly (K)')
    ax.set_ylim(ylim)
    if show_legend:
        ax.legend(loc=legend_loc)
            
    
def barplot_monthly(ax, exceedances, percentile=None, show_bias=True, yticks=[0, 5, 10], ylim=(0, 15), color=red):
    daysinmonth = [exceedances.sel(time=exceedances['time.month'] == month)['time.daysinmonth'].values[0] 
                   for month in range(1, 13)] 
    month_bins = np.cumsum([0] + daysinmonth)  # needed for alignment of x-axis

    ax.bar(
        month_bins[:-1], 
        (exceedances > 0).groupby('time.month').mean() * 100, 
        width=np.array(daysinmonth) - 1, 
        align='edge', 
        color=color,
    )
    if percentile is not None:
        ax.axhline(100 - percentile, color='k', ls='--')
        print('Annual mean bias: {:.1f}%'.format(calc_bias(exceedances, percentile).item()))    
        
    set_xticks(ax)
    ax.set_ylabel('Freq. (%)')
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    
    if percentile is not None and show_bias:
        for doy, freq in zip(.5*(month_bins[1:] + month_bins[:-1]), (exceedances > 0).groupby('time.month').mean()):
            bias = np.around(calc_bias(freq, percentile))
            ax.text(doy, .05, f'{bias:+.0f}%', va='bottom', ha='center', fontsize='large')
            
    
def barplot_monthly_difference(ax, exceedances, exceedances_ref, color=red, show_bias=True):
    daysinmonth = [exceedances.sel(time=exceedances['time.month'] == month)['time.daysinmonth'].values[0] 
                   for month in range(1, 13)] 
    month_bins = np.cumsum([0] + daysinmonth)  # needed for alignment of x-axis
    
    exc = (exceedances > 0).groupby('time.month').mean() 
    ref = (exceedances_ref > 0).groupby('time.month').mean() 

    ax.bar(
        month_bins[:-1] + 4, 
        exc * 100, 
        width=np.array(daysinmonth) - 8, 
        align='edge', 
        color=color,
    )
    
    if show_bias:
        for doy, change in zip(.5*(month_bins[1:] + month_bins[:-1]), (exc - ref) / ref):
            change = np.around(change * 100)
            ax.text(doy, .05, f'{change:+.0f}%', va='bottom', ha='center', fontsize='large')
            
    print('Mean relative difference: {:.1f}%'.format(
        ((exceedances > 0).mean() - (exceedances_ref > 0).mean()) / 
         (exceedances_ref > 0).mean() * 100))
    
    
def barplot_daily(ax, frequencies, percentile, yticks=[0, 5, 10], ylim=(0, 15)):
    ax.axhline(100 - percentile, color='k', ls='--', zorder=10)
    
    ax.bar(
        np.arange(len(frequencies)), 
        frequencies * 100, 
        width=1, 
        align='edge', 
        color=red,
        zorder=99
    )
    
    set_xticks(ax)
    ax.set_ylabel('Freq. (%)')
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)