class Era5_Conf:
    def __init__(self, version='c88'):
        if version == "v88":
            self.var_needed_dic = dict(
                t2m='2m_temperature',
                u10m='10m_u_component_of_wind',
                v10m='10m_v_component_of_wind',
                msl='mean_sea_level_pressure',
                tp='total_precipitation',
                d2m='2m_dewpoint_temperature',
                sst='sea_surface_temperature',
                u100m='100m_u_component_of_wind',
                v100m='100m_v_component_of_wind',
                lcc='low_cloud_cover',
                mcc='medium_cloud_cover',
                hcc='high_cloud_cover',
                tcc='total_cloud_cover',
                ssr='surface_net_solar_radiation',
                ssrd='surface_solar_radiation_downwards',
                fdir='total_sky_direct_solar_radiation_at_surface',
                ttr='top_net_thermal_radiation',
                mdts='mean_direction_of_total_swell',
                mdww='mean_direction_of_wind_waves',
                mpts='mean_period_of_total_swell',
                mpww='mean_period_of_wind_waves',
                shts='significant_height_of_total_swell',
                shww='significant_height_of_wind_waves',
            )

            self.var_needed_pl_dic = dict(
                z='geopotential',
                t='temperature',
                u='u_component_of_wind',
                v='v_component_of_wind',
                q='specific_humidity',
            )

            self.levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
            
            
        elif version == "sais.2025":
            self.var_needed_dic = dict(
                )

            self.var_needed_pl_dic = dict(
                z='geopotential',
                t='temperature',
                u='u_component_of_wind',
                v='v_component_of_wind',
                q='specific_humidity',
                ciwc="specific_cloud_ice_water_content",
                clwc="specific_cloud_liquid_water_content",
                crwc="specific_rain_water_content",
                cswc="specific_snow_water_content"
            )

            self.levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        else:
            raise ValueError("Unsupported version")