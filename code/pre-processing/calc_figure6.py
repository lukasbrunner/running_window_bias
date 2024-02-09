import os
import xarray as xr
import logging

from core_functions import delete_short_periods
from io_functions import get_filename, model_names, DATA_PATH
from logger_functions import set_logger

logger = logging.getLogger(__name__)
set_logger()

ds_list = []
for model in model_names:
    logger.info(model)
    ds = xr.open_dataset(get_filename(model, startyear_test=2071, endyear_test=2100))
    ds = ds.drop_vars('height', errors='ignore')
    ds_agg = xr.Dataset()
    for varn in [
        'exceedances', 'exceedances_deseasonalized', 
        'exceedances_test', 'exceedances_test_deseasonalized']:
        timen = 'time'
        if 'test' in varn:
            timen = 'time2'
            
        # --- pure time aggregation ---
        ds_agg[varn] = (ds[varn] > 0).mean(timen)
        # restrict to periods >= 3 days
        ds_agg[varn + '_3d'] = delete_short_periods(ds[varn] > 0).mean(timen)        
        # restrict to periods >= 6 days
        ds_agg[varn + '_6d'] = delete_short_periods(ds[varn] > 0, 6).mean(timen)
        
        # --- restrict to extended summer ---
        tmp = (ds[varn] > 0).where(
            ((ds['{}.month'.format(timen)].isin([5, 6, 7, 8, 9])) & (ds['lat'] > 0) |
             (ds['{}.month'.format(timen)].isin([11, 12, 1, 2, 3])) & (ds['lat'] < 0)))
        ds_agg[varn + '_extended-summer'] = tmp.mean(timen)
        ds_agg[varn + '_extended-summer_3d'] = delete_short_periods(tmp).mean(timen)
        ds_agg[varn + '_extended-summer_6d'] = delete_short_periods(tmp, 6).mean(timen)
        
        # --- restrict to summer ---
        tmp = (ds[varn] > 0).where(
            ((ds['{}.month'.format(timen)].isin([6, 7, 8])) & (ds['lat'] > 0) |
             (ds['{}.month'.format(timen)].isin([12, 1, 2])) & (ds['lat'] < 0)))
        ds_agg[varn + '_summer'] = tmp.mean(timen)
        ds_agg[varn + '_summer_3d'] = delete_short_periods(tmp).mean(timen)
        ds_agg[varn + '_summer_6d'] = delete_short_periods(tmp, 6).mean(timen)
        
        # --- restrict to winter ---
        tmp = (ds[varn] > 0).where(
            ((ds['{}.month'.format(timen)].isin([12, 1, 2])) & (ds['lat'] > 0) |
             (ds['{}.month'.format(timen)].isin([6, 7, 8])) & (ds['lat'] < 0)))
        ds_agg[varn + '_winter'] = tmp.mean(timen)
        ds_agg[varn + '_winter_3d'] = delete_short_periods(tmp).mean(timen)
        ds_agg[varn + '_winter_6d'] = delete_short_periods(tmp, 6).mean(timen)
        
        # --- restrict to spring ---
        tmp = (ds[varn] > 0).where(
            ((ds['{}.month'.format(timen)].isin([3, 4, 5])) & (ds['lat'] > 0) |
             (ds['{}.month'.format(timen)].isin([9, 10, 11])) & (ds['lat'] < 0)))
        ds_agg[varn + '_spring'] = tmp.mean(timen)
        ds_agg[varn + '_spring_3d'] = delete_short_periods(tmp).mean(timen)
        ds_agg[varn + '_spring_6d'] = delete_short_periods(tmp, 6).mean(timen)
        
        # --- restrict to fall ---
        tmp = (ds[varn] > 0).where(
            ((ds['{}.month'.format(timen)].isin([9, 10, 11])) & (ds['lat'] > 0) |
             (ds['{}.month'.format(timen)].isin([3, 4, 5])) & (ds['lat'] < 0)))
        ds_agg[varn + '_fall'] = tmp.mean(timen)      
        ds_agg[varn + '_fall_3d'] = delete_short_periods(tmp).mean(timen)
        ds_agg[varn + '_fall_6d'] = delete_short_periods(tmp, 6).mean(timen)
        
    ds_list.append(ds_agg)
    
ds = xr.concat(ds_list, dim='model')   
ds.to_netcdf(os.path.join(DATA_PATH, 'figure6.nc'))