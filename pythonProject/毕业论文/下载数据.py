#!/usr/bin/python3
import cdsapi
import os
import pandas as pd
import numpy as np

c = cdsapi.Client()

var_list = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
            '2m_temperature', 'land_sea_mask', 'mean_sea_level_pressure',
            'sea_ice_cover', 'sea_surface_temperature', 'skin_temperature',
            'snow_depth', 'soil_temperature_level_1', 'soil_temperature_level_2',
            'soil_temperature_level_3', 'soil_temperature_level_4', 'surface_pressure',
            'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3',
            'volumetric_soil_water_layer_4']

start_date = '20070101'
end_date = '20071231'
start_year = int(start_date[:4])
end_year = int(end_date[:4])

year_list = np.arange(start_year, end_year+1)
year_list = year_list.astype(str)

for iyear in year_list:
    path = './' + iyear
    if os.path.exists(path):
        print("file exist")
    else:
        os.mkdir(path)
    year_start_date = iyear+"0101"
    year_end_date   = end_date if int(end_date) < int(iyear)*10000+1231 else iyear+"1231"
    date_list = pd.date_range(year_start_date, year_end_date).strftime("%Y%m%d").tolist()
    for idate in date_list:
        date_name = path + "/" + idate + "-sfc.grib"
        if os.path.exists(date_name):
            continue

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'grib',
                'variable': var_list,
                'year': iyear,
                'month': idate[4:6],
                'day': idate[6:8],
                'time': [
                    '00:00', '06:00', '12:00',
                    '18:00',
                ],
                },
                date_name)
print('ok')