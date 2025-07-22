"""
| Copyright (c) 2025
| Deutsches Zentrum für Luft- und Raumfahrt e.V. (DLR)
| Linder Höhe
| 51147 Köln

:Authors: Judith Heusel, Michael Roth, Keivan Kiyanfar, Erik Hedberg

:Version: 1.4.0

:Project: railpos

License
-------
This software is released under the 3-Clause BSD License.
Details about the license can be found in the LICENSE file.

Description
-------
Functions for preparing sensor dataframes.
"""

import pandas as pd
import geopandas as gpd


def df2gdf(df: pd.DataFrame, lon_column: str = 'lon', lat_column: str = 'lat', epsg: int = 4326,
           decimation_factor: int = 1):
    """
    Convert a DataFrame which contains geospatial information into a GeoDataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame which will be converted. Must contain geospatial information from which geometries are created.
        Must contain columns specified by lon_column and lat_column. Default is geometry creation using longitudinal and
        lateral coordinates.
    lon_column : str, optional
        Specifies the column with longitudinal coordinates. The default is 'lon'.
    lat_column : str, optional
        Specifies the column with lateral coordinates. The default is 'lat'.
    epsg : int, optional
        Specifies the spatial reference system to create the geometries (to which the values in lon_column and
        lat_column relate). The default is 4326.
    decimation_factor : int, optional
        Factor by which the data is reduced. The default is 1.

    Returns
    -------
    gdf : gpd.GeoDataFrame
        A GeoDataFrame with geometries created from the values in lon_column and lat_column.

    Raises:
        ValueError: If either lon_column or lat_column are not contained in the columns of df or if epsg is not an
        integer.



    """
    if lon_column not in df.columns or lat_column not in df.columns:
        raise ValueError('Either lon_column or lat_column not contained in columns of df.')

    if not isinstance(epsg, int):
        raise ValueError('EPSG must be an integer.')

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_column], df[lat_column]), crs=f"epsg:{epsg}")

    if decimation_factor > 1:
        gdf = gdf.iloc[::int(decimation_factor), :]
        gdf.reset_index(inplace=True)

    return gdf


if __name__ == '__main__':
    pass
