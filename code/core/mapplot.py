#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2023 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || l.brunner@univie.ac.at

Abstract: Functions to create mapplots of frequencies.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Polygon, Patch
import seaborn as sns

from core.core_functions import spatial_aggregation

plt.rcParams['hatch.linewidth'] = .5
plt.rcParams['hatch.color'] = 'k'

proj=ccrs.Robinson()


def antimeridian_pacific(lon: 'xr.DataArray') -> bool:
    """Returns True if the antimeridian is in the Pacific (i.e. longitude runs
    from -180 to 180."""
    if lon.min() < 0 and lon.max() < 180:
        return True
    if lon.min() >= 0 and lon.max() >= 180:
        return False
    raise ValueError('Could not establish antimeridian')
    
    
def flip_antimeridian(
        dataarray: 'xr.DataArray',
        to: str='Pacific',
        lonn: str='lon') -> 'xr.DataArray':
    """
    Flip the antimeridian (i.e. longitude discontinuity) between Europe
    (i.e., [0, 360)) and the Pacific (i.e., [-180, 180)).

    Parameters:
    - dataarray : xarray.DataArray
    - to : string, {'Pacific', 'Europe'}
      * 'Europe': Longitude will be in [0, 360)
      * 'Pacific': Longitude will be in [-180, 180)
    - lonn: string, optional

    Returns:
    dataarray_flipped : xarray.DataArray
    """
    lon = dataarray[lonn]
    lon_attrs = lon.attrs

     # already correct, do nothing
    if to.lower() == 'europe' and not antimeridian_pacific(lon):
        return dataarray
    if to.lower() == 'pacific' and antimeridian_pacific(lon):
        return dataarray

    if to.lower() == 'europe':
        dataarray = dataarray.assign_coords(**{lonn: (lon % 360)})
    elif to.lower() == 'pacific':
        dataarray = dataarray.assign_coords(**{lonn: (((lon + 180) % 360) - 180)})
    else:
        errmsg = "to has to be one of {'Europe', 'Pacific'} not {}".format(to)
        raise ValueError(errmsg)

    idx = np.argmin(dataarray[lonn].values)
    dataarray = dataarray.roll(**{lonn: -idx}, roll_coords=True)
    dataarray[lonn].attrs = lon_attrs
    return dataarray


def get_defaults(levels): 
    if levels is None:
        return {}
    
    if np.max(levels) == 5 and np.min(levels) == -30:  # default: only one positive level
        cmap = plt.cm.get_cmap('RdBu_r', 14)
        colors = list(cmap(np.arange(14)))
        colors[7] = colors[8] # make the only red a bit darker
        colors[8] = colors[9] # make the only red a bit darker
        return {'extend': 'min', 'levels': levels, 'colors': colors}
    
    if np.min(levels) >= 0:  # plot frequency not bias
        cmap = plt.cm.get_cmap('RdBu_r', len(levels) * 2)
        colors = list(cmap(np.arange(len(levels) * 2)))
        colors[len(levels)-1] = colors[len(levels)-2] # make the only blue a bit darker

        return {'extend': 'both', 'levels': levels, 'colors': colors[len(levels)-1:]}
    
    cmap = plt.cm.get_cmap('RdBu_r', len(levels) + 1)
    colors = list(cmap(np.arange(len(levels) + 1)))
    return {'extend': 'both', 'levels': levels, 'colors': colors}


def plot_map(da, equator_line=False, add_colorbar=True, levels=np.arange(-30, 6, 5), missing_grey=False, **kwargs):
    kwargs_default = get_defaults(levels)
    
    if add_colorbar:
        kwargs_default['cbar_kwargs'] = {
            'pad': .01, 
            # 'magic' ratio via: https://stackoverflow.com/a/26720422/2699929
            'fraction': 0.046,
            'label': ''
        }
        
    kwargs_default.update(kwargs)
    
    # NOTE: we need to convert to [-180, 180], otherwise cmap.set_bad will not work properly
    da = flip_antimeridian(da)
    
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(8, 4))
    map_ = da.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), **kwargs_default)
    if missing_grey:
        map_.cmap.set_bad('grey')
      
    if equator_line:  # to hilight hemispheric separation for extended summer
        ax.axhline(0, color='k', ls='--')
    ax.coastlines()
    ax.gridlines(color='gray', lw=.5, ls=':') 
    
    print('Mean (0/5/95/100 perc): {:.1f}% ({:.1f}%/{:.1f}%/{:.1f}%/{:.1f}%)'.format(
        spatial_aggregation(da, ignore_inf=True),
        spatial_aggregation(da, 0, ignore_inf=True),
        spatial_aggregation(da, 5, ignore_inf=True),
        spatial_aggregation(da, 95, ignore_inf=True),
        spatial_aggregation(da, 100, ignore_inf=True),
    ))
    return fig, ax 


def plot_boxes(ax, da, boxes, colors='orangered', texts=None, widen=1.5):
    boxes = np.atleast_1d(boxes)
    if isinstance(colors, str):
        colors = [colors] * len(boxes)
        
    dx = 1.25 + widen
    for idx, (lat, lon) in enumerate(boxes):
        lat, lon = da.sel(lat=lat, method='nearest')['lat'].item(), da.sel(lon=lon, method='nearest')['lon'].item()
        print('lat={}, lon={}'.format(lat, lon))
        poly = Polygon(
            [[lon-dx, lat-dx], [lon+dx, lat-dx], 
             [lon+dx, lat+dx], [lon-dx, lat+dx]], 
            transform=ccrs.PlateCarree(), 
            fill=False,
            # facecolor=colors[idx],
            edgecolor=colors[idx],
            lw=2, 
            closed=True,
            zorder=99,
        )
        ax.add_patch(poly)
        if texts is not None:
            ax.text(lon+dx+.5, lat, texts[idx], va='center', transform=ccrs.PlateCarree())
            

def plot_hatching(ax, da, fraction=.8, min_value=0):
    # Def. robust: at least X% of models agree on the sign of the change
    da = xr.where(
        (((da > 0).mean('model') > fraction) |
        ((da < 0).mean('model') > fraction)) &
        (np.abs(da.mean('model')) > min_value), 
        True, False)

    stippling = flip_antimeridian(da)  # avoid artefact at 0E

    ax.contourf(
        da['lon'], da['lat'], da,
        transform=ccrs.PlateCarree(), 
        levels=[.5, 1.5],
        colors='none',
        hatches=['....'],
    )

    legend_elements = [
        Patch(facecolor='none', alpha=.99, hatch='....', label='Robust'),
    ]
    ax.legend(handles=legend_elements, loc=(.86, 0.), handlelength=1, fontsize='small')
        
    print('Fraction robust: {:.1%}'.format(spatial_aggregation(da).item()))