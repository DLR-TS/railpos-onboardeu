"""
| Copyright (c) 2025
| Deutsches Zentrum für Luft- und Raumfahrt e.V. (DLR)
| Linder Höhe
| 51147 Köln

:Authors: Judith Heusel, Keivan Kiyanfar, Michael Roth, Erik Hedberg

:Version: 1.4.0

:Project: railpos

License
-------
This software is released under the 3-Clause BSD License.
Details about the license can be found in the LICENSE file.

Description
Helper functions for rail vehicle positioning, data reading and saving within the mFUND project OnboardEU.
-------
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import h5py
import folium
import matplotlib.colors as mcolors
from typing import Union

from railpos.helpers import integrate_signal
from railpos.visualization import folium_layer_control
from railpos.railwaypaths import RailwayPath


def get_measurement_data(path_to_file: Union[str, os.PathLike], index_journey: int):
    """
    Function to read GNSS and IMU data from a HDF5 file with a specific structure used to provide data in the project.
    The HDF file contains groups, each group containing the data of a journey. Each group has two datasets, named GNSS
    and IMU, for the GNSS and IMU sensor data, respectively.

    Parameters
    ----------
    path_to_file : str or path like
        File path of the .h5 file.
    index_journey : int
        Number of the journey that will be read from the file.

    Returns
    -------
    df_gnss : pd.DataFrame
        Contains the GNSS data of the measurement system for this journey.
    df_imu : pd.DataFrame
        Contains the IMU data of the measurement system for this journey.
    attrs_gnss: dict
        The attributes of the GNSS sensor data.
    attrs_imu: dict
        The attributes of the IMU sensor data.

    """

    with h5py.File(path_to_file, 'r') as f:
        path_imu = 'journey_' + str(index_journey).zfill(2) + '/IMU'
        path_gnss = 'journey_' + str(index_journey).zfill(2) + '/GNSS'
        df_gnss = pd.DataFrame(np.array(f.get(path_gnss)))
        df_imu = pd.DataFrame(np.array(f.get(path_imu)))
        attrs_gnss = eval(f[path_gnss].attrs['GNSS'])
        attrs_imu = eval(f[path_imu].attrs['IMU'])
        f.close()
    return df_gnss, df_imu, attrs_gnss, attrs_imu


def get_gdf_projected_points(gdf: gpd.GeoDataFrame, rpath: RailwayPath, columns_to_keep: list):
    """
    Create a GeoDataFrame containing the projections of GNSS data to the geometry of a railway path.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Contains GNSS data, the geometries are given by the positions of the GNSS data.
    rpath : RailwayPath
        The RailwayPath object of the railway path to project on.
    columns_to_keep : list
        Names of the columns which are transferred to the GeoDataFrame containing the projected data.

    Returns
    -------
    gdf_projected : gpd.GeoDataFrame
        Geometries are given by the projected geometries of gdf to the geometry of the RailwayPath object.
        Additional information are given by the projection errors and the distances with respect to the path.

    """
    geoms = []
    data = np.zeros((len(gdf), 2))
    for i, pt in enumerate(gdf.geometry):
        pt_proj, data[i, 0], data[i, 1] = rpath.points2path(pt)
        geoms.append(pt_proj)
    gdf_projected = gpd.GeoDataFrame(data=data, geometry=geoms, columns=['projection_error', 'distance_on_path'])
    for col in columns_to_keep:
        gdf_projected[col] = gdf[col].values
    return gdf_projected


def georef_plot_inputs(df_imu: pd.DataFrame, df_gnss: pd.DataFrame, fig_num: int = 0, **kwargs):
    """
    Plot input sensor data (GNSS and IMU). Specific for the data and naming in this project.

    Parameters
    ----------
    df_imu : pd.DataFrame
        Contains IMU time information and acceleration measurement values (columns: times and acc_x).
    df_gnss : pd.DataFrame
        Contains GNSS time information, number of satellites and speed.
    fig_num : int, optional
        Number of the first figure. The default is 0.
    kwargs: Keyword args to plt.subplots().

    Returns
    -------
    fig : plt.Figure
        Plot of GNSS speed, number of satellites and integrated IMU acceleration.
    fig_2 : plt.Figure
        Plot of IMU acceleration and compensated values by journey mean and median.

    """
    plt.close(fig_num)
    fig, ax = plt.subplots(num=fig_num, **kwargs)
    ax.plot(df_gnss.times, df_gnss.speed, marker='.', label='GNSS Speed')
    ax.plot(df_imu.times, integrate_signal(df_imu.acc_x, df_imu.times), label='Integrated IMU Acceleration')
    ax.plot(df_gnss.times, df_gnss["nsat"], label='Number of Satellites')
    ax.set_xlabel('Time in s')
    ax.set_ylabel('Speed in m/s')
    ax.legend()
    fig.tight_layout()

    plt.close(fig_num + 1)
    fig_2, ax = plt.subplots(num=fig_num + 1, **kwargs)
    ax.plot(df_imu.times, df_imu.acc_x, label='IMU Acceleration')
    ax.plot([df_imu.times.min(), df_imu.times.max()], df_imu.acc_x.mean() * np.ones(2), label='mean')
    ax.plot([df_imu.times.min(), df_imu.times.max()], df_imu.acc_x.median() * np.ones(2), label='median')
    ax.plot(df_imu.times, df_imu.acc_x - df_imu.acc_x.median(), label='Compensated acceleration')
    ax.legend()
    ax.set_xlabel('Time in s')
    ax.set_ylabel('Acceleration in m/(s**2)')
    fig_2.tight_layout()
    return fig, fig_2


def add_path_hypotheses_to_folium(folium_map: folium.folium.Map, rpaths: list, add_layer_control: bool = True):
    """
    Add the geometries of a set of RailwayPath objects to a given folium map.

    Parameters
    ----------
    folium_map : folium.folium.Map
        Input map (e.g. visualising measurement data or the railway network).
    rpaths : list
        A list of RailwayPath objects whose geometries should be visualised.
    add_layer_control : bool, optional
        If True, folium.LayerControl is added to the folium map.
        This can only be done once, so this should be False if something will be added later to the map.
        The default is True.

    Returns
    -------
    folium_map : folium.folium.Map
        Input folium map supplemented by the geometries of the RailwayPath objects given in rpaths.

    """
    color_list = list(mcolors.TABLEAU_COLORS.values())
    for i, rpath in enumerate(rpaths):
        rpath.railway_map.geo_df.iloc[rpath.tracks].explore(m=folium_map, name='Path %s' % i, color=color_list[i % 10])
    if add_layer_control:
        folium_layer_control(folium_map)
    return folium_map


def georef_plot_path_projection_errors(projection_errors_matrix: np.ndarray, **kwargs):
    """
    Plot projection errors of GNSS data to different railway paths.

    Parameters
    ----------
    projection_errors_matrix : np.ndarray
        Matrix of projection errors of GNSS data for a set of RailwayPath objects. Shape: (nr_of_paths,
        nr_of_GNSS_points).
    **kwargs :
        Keyword args to be passed to plt.subplots.

    Returns
    -------
    fig : plt.Figure
        Plot of the projection errors.

    """
    fig, ax = plt.subplots(**kwargs)
    plt.pcolormesh(projection_errors_matrix)
    cbar = plt.colorbar()
    cbar.set_label('Projection error in m')
    ax.set_xlabel('GNSS index')
    ax.set_ylabel('Path hypothesis')
    fig.tight_layout()
    return fig


def struct_table_result(df: pd.DataFrame):
    """
    Returns structured data to be saved in hdf5

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame which will be saved in a H5 dataset.

    Returns
    -------
    res : np.ndarray
        Structured data that can be saved as dataset.

    """
    dtypes = []
    for column in df.columns:
        data_type = h5py.special_dtype(vlen=str) if df[column].dtype == object else df[column].dtype
        dtypes.append((column, data_type))
    structured_dtype = np.dtype(dtypes)
    res = np.zeros(df.shape[0], dtype=structured_dtype)
    for column in df.columns:
        res[column] = df[column].values
    return res


def georef_write_results(gdf_georef: gpd.GeoDataFrame, output_path: str, group_path: str = None,
                         dset_name: str = 'georeferencing_result'):
    """
    Writes the results of the geo-referencing in an .h5 file.

    Parameters
    ----------
    gdf_georef : gpd.GeoDataFrame
        Contains the geo-referencing results.
    output_path : str or path object
        File path of the output file. Must end with .h5
    group_path: str, optional
        Path of the group to which the dataset is added. If None, dataset is added at highest level.
        The default is None.
    dset_name: str, optional
        Name of the dataset in the .h5 file. The default is 'georeferencing_result'.

    Returns
    -------
    None.

    """
    gdf_temp = gdf_georef.copy()
    gdf_temp.drop(columns=['geometry'], inplace=True)
    gdf_temp = pd.DataFrame(gdf_temp)
    structured_array = struct_table_result(df=gdf_temp)
    with h5py.File(output_path, 'a') as f:
        # Create a dataset from the structured array
        if group_path is not None:
            f.create_group(group_path)
            f[group_path].create_dataset(dset_name, data=structured_array, compression='gzip')
        else:
            f.create_dataset(dset_name, data=structured_array, compression='gzip')


if __name__ == '__main__':
    pass
