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
Provides a class and functionalities for handling maps of railway infrastructure.
"""

import os
import warnings
import numpy as np
import geopandas as gpd
import shapely as shp
import networkx as nx
from typing import Union, List


class RailwayMap:
    """
    Class for handling geometries of a railway network. Provides functionalities to find close tracks to data, to get
    geometry properties of the tracks, to find paths in the railway network that connect two arbitrary tracks and to
    find railway paths in the network given GNSS data (offline positioning).
    Tracks are referred to by integer indices.

    Parameters
    ----------
    geo_df : gpd.GeoDataFrame
        A gpd.GeoDataFrame which contains track geometries and connectivity information.

    Attributes
    ----------
    G_directed_connections : nx.DiGraph
        A directed graph which contains information about connectivity of oriented tracks. Nodes are given by pairs of
        tracks and orientations. Edges are given by valid connections, e.g, (i, 1) is connected to (j, -1) if the end
        of track i is connected to the end of track j. This means that a vehicle which moves along track i with
        increasing track distance can enter track j in direction -1, e.g. move along track j with decreasing track
        distance without stopping.
    geo_df : gpd.GeoDataFrame
        The gpd.GeoDataFrame containing the track geometries.


    """
    def __init__(self, geo_df: gpd.GeoDataFrame):
        """

        Parameters
        ----------
        geo_df : gpd.GeoDataFrame
            Must contain the tracks of the railway network as geometries. Must already contain information about the
            connectivity of the tracks, i.e., if a vehicle can enter track i from track j. This information must be
            contained in the columns 'connected_start' and 'connected_end'. If track i is contained in 'connected_start'
            of track j, then track i is connected to the start (first boundary or geom.interpolate(0)) of track j.

        Returns
        -------
        None.

        Raises:
            ValueError: If one of the columns 'connected_start' or 'connected_end' is not contained in geo_df.

        """
        self.G_directed_connections = None
        self.geo_df = geo_df
        if 'connected_end' not in geo_df.columns or 'connected_start' not in geo_df.columns:
            raise ValueError('Input must contain columns named connected_start and connected_end.')

        # ensure that the connection columns contain lists and not strings
        if isinstance(self.geo_df['connected_end'].iloc[0], str):
            self.geo_df['connected_end'] = [eval(conn_str) for conn_str in self.geo_df['connected_end']]
            self.geo_df['connected_start'] = [eval(conn_str) for conn_str in self.geo_df['connected_start']]

        # simplify geometries to solve some numerical issues
        for i in self.geo_df.index:
            geom = self.geo_df.loc[i, 'geometry']
            self.geo_df.loc[i, 'geometry'] = shp.simplify(geom, tolerance=0.001)

    @classmethod
    def from_geopackage(cls, filename: Union[str, os.PathLike]):
        """
        Read map data from a geopackage and create a RailwayMap object.

        Parameters
        ----------
        filename : str or path-like
            File path for reading the map data.


        Returns
        -------
        RailwayMap
            A RailwayMap created from the file content.

        """
        gdf = gpd.read_file(filename)

        return RailwayMap(gdf)

    def check_connection_symmetry(self):
        """
        Check the connections in the gpd.GeoDataFrame of the railway network.
        If track j is listed in the connections to track i, then i must be listed in the connections of track j.

        Returns
        -------
        symmetry_flag : bool
            If True, connection symmetry is given.

        """
        symmetry_flag = True

        # check connections
        for i in range(len(self.geo_df)):
            connected_to_i = self.geo_df['connected_start'].iloc[i] + self.geo_df['connected_end'].iloc[i]

            for j in connected_to_i:
                connected_to_j = self.geo_df['connected_start'].iloc[j] + self.geo_df['connected_end'].iloc[j]
                if i not in connected_to_j:
                    symmetry_flag = False

        return symmetry_flag

    def oriented_track_geometry(self, track_identifier: int, direction: bool = True):
        """
        Get the oriented track geometry for a track.

        Parameters
        ----------
        track_identifier : int
            Track identifier (index) of required track.
        direction : bool or int (+-1)
            Orientation of the track (True, False).

        Returns
        -------
        shp.LineString
            Oriented track geometry.

        """
        direction = bool(direction)
        if direction:
            return self.geo_df.geometry.iloc[track_identifier]
        else:
            return self.geo_df.geometry.iloc[track_identifier].reverse()

    def track_coordinates(self, track_identifier: int, direction: bool = True):
        """

        Parameters
        ----------
        track_identifier : int
            Track identifier (index) of required track.
        direction : bool or int (+-1)
            Orientation of the track (True, False).

        Returns
        -------
        np.ndarray
            Coordinates of the track (according to orientation).

        """
        return np.array(self.oriented_track_geometry(track_identifier=track_identifier, direction=direction).coords)

    def track_length(self, track_identifier: int):
        """
        Get the length of a given track

        Parameters
        ----------
        track_identifier : int
            Track identifier (index) of the required track.

        Returns
        -------
        float
            Length of the track.
        """
        return self.geo_df.geometry.iloc[track_identifier].length

    def distances_to_tracks(self, geom):
        """
        Get the distances of an input geometry to all tracks of the network.

        Parameters
        ----------
        geom : shapely geometry

        Returns
        -------
        distances : np.ndarray
            Distances to all tracks of the input.

        """
        distances = self.geo_df.distance(geom)
        return distances

    def closest_track(self, p):
        """
        Get the index of the closest track of the network to a geometry.

        Parameters
        ----------
        p : shp geometry.

        Returns
        -------
        int
            Track(s) of closest track(s) to the geometry.

        """
        distances = self.distances_to_tracks(p)
        # calculate min distance to get all tracks in the closest distance and not only the first one
        min_dist = np.nanmin(distances)
        closest_tracks = np.where(distances == min_dist)[0]
        return self.geo_df.index[closest_tracks]

    def create_graph_directed_connections(self, tolerance: float = 0.01):
        """
        Create a directed graph whose nodes are pairs of (track identifier, orientation) and an edge between two nodes
        is present if the two tracks are connected and a transition from the first track to the second track is possible
        without stopping (direction change) along the given orientations.
        This means that there is an edge from (track i, orientation o_i) to (track j, orientation o_j) if a vehicle
        that is moving along track i according to o_i (orientation = True or 1 means it moves into the direction of
        increasing track distances) can transit from i to j and move there according to o_j without changing the
        direction (without stopping).
        The graph is used to find valid paths in the railway network from one track to another.

        Parameters
        ----------
        tolerance: float, optional
        Tolerance when determining if track boundaries match. The default is 0.01.

        Returns
        -------
        None.

        """

        self.G_directed_connections = nx.DiGraph()
        nodes = []
        for track_identifier in range(len(self.geo_df)):
            nodes.extend([(track_identifier, 1), (track_identifier, -1)])
        self.G_directed_connections.add_nodes_from(nodes)

        for track_identifier in range(len(self.geo_df)):
            track = self.geo_df.geometry.iloc[track_identifier]
            start, end = track.boundary.geoms
            node1 = (track_identifier, 1)
            node2 = (track_identifier, -1)

            connected_to_end = self.geo_df.connected_end.iloc[track_identifier]
            for element in connected_to_end:
                conn_track = self.geo_df.geometry.iloc[element]
                start2, end2 = conn_track.boundary.geoms

                if end.equals_exact(start2, tolerance):
                    self.G_directed_connections.add_edge(node1, (element, 1))

                if end.equals_exact(end2, tolerance):
                    self.G_directed_connections.add_edge(node1, (element, -1))

            connected_to_start = self.geo_df.connected_start.iloc[track_identifier]
            for element in connected_to_start:
                conn_track = self.geo_df.geometry.iloc[element]
                start2, end2 = conn_track.boundary.geoms

                if start.equals_exact(start2, tolerance):
                    self.G_directed_connections.add_edge(node2, (element, 1))

                if start.equals_exact(end2, tolerance):
                    self.G_directed_connections.add_edge(node2, (element, -1))

    def tracks_near_data(self, data: Union[shp.geometry, List[shp.geometry], gpd.GeoDataFrame], distance: float,
                         margin=None, interpolate=True):
        """
        Determine all tracks near data points.


        Parameters
        ----------
        data : Single shapely geometries, list of shapely geometries or geopandas.GeoDataFrame.
            Geometries of (GNSS) data for which all tracks in a certain distance are determined.
        distance : float
            Maximal distance to the data within which tracks will be selected.
        margin : float, optional
            Will draw a bounding box around the data points and add the margin to each bounding.
            Tracks will be selected if they cross the bounding box. The default is None.
        interpolate : bool, optional
            If True: Data points will be converted to a linestring, if the data contains only shp.Point objects.
            The default is True.

        Returns
        -------
        tracks_near_data : list
            A list of all indices referring to a track that is within the given distance to the data.

        """
        if isinstance(data, list):
            geoms = data
        elif isinstance(data, gpd.GeoDataFrame) or isinstance(data, gpd.GeoSeries):
            geoms = [geom for geom in data.geometry]
        elif isinstance(data, shp.Point) or isinstance(data, shp.LineString):
            geoms = [data]
        elif isinstance(data, np.ndarray):
            data = shp.Point(data)
            geoms = [data]

        tracks_near_data = list()

        if interpolate:
            types = list(set([type(geom) for geom in geoms]))
            if len(types) == 1:
                if types[0] == shp.Point and len(geoms) > 1:
                    interpolated = shp.LineString(geoms)
                    geoms = [interpolated]

        if margin is not None:
            xmin, ymin, xmax, ymax = shp.total_bounds(geoms)
            xmin, ymin = xmin - margin, ymin - margin
            xmax, ymax = xmax + margin, ymax + margin

            nodes = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
            nodes.append(nodes[0])
            nodes = tuple(nodes)
            poly = shp.Polygon(nodes)
            tracks_near_data = [i for i in range(len(self.geo_df)) if
                                shp.distance(poly, self.geo_df.geometry.iloc[i]) == 0]

        close_tracks = [self.geo_df.loc[p.dwithin(self.geo_df.geometry, distance=distance)].index for p in geoms]
        for tr in close_tracks:
            tracks_near_data.extend(list(tr))
        tracks_near_data = list(set(tracks_near_data))

        return tracks_near_data

    def subgraph_around_data(self, data: Union[shp.geometry, List[shp.geometry], gpd.GeoDataFrame], distance: float,
                             margin: float = None, interpolate: bool = True):
        """
        Get a subgraph of self.G_directed_connections which contains nodes and edges generated from all tracks
        near the given data.

        Parameters
        ----------
        data : Single shapely geometries, list of shapely geometries or geopandas.GeoDataFrame.
            Geometries of (GNSS) data.
        distance : float
            Max distance of a track to the data points to be contained in the output.
        margin : float, optional
            Will draw a bounding box around the data points and add the margin to each bounding.
            Tracks will be selected if they cross the bounding box. The default is None.
        interpolate : bool, optional
            If True, the data will be converted to a linestring in case it data consists only of shp.Points.
            The default is True.

        Returns
        -------
        nx.DiGraph()
            Subgraph containing relevant connections (involving all tracks close to the data).

        """

        if not self.G_directed_connections:
            self.create_graph_directed_connections()
        tracks_near_data = self.tracks_near_data(data=data, distance=distance, margin=margin, interpolate=interpolate)
        nodes_area = [(track, 1) for track in tracks_near_data]
        nodes_area.extend([(track, -1) for track in tracks_near_data])
        return self.G_directed_connections.subgraph(nodes_area)

    def valid_paths_for_data_points(self, data, distance: float = 60, margin: bool = None, interpolate: bool = True,
                                    cutoff: int = None, distance_start_end: float = 15):
        """
        Determine all valid paths in the railway network that connect tracks close to the first and the last data point.

        Parameters
        ----------
        data : list of shapely geometries or geopandas.GeoDataFrame.
            Input data (geometries/positions).
        distance : float, optional
            Only tracks within this distance to the data will be considered. The default is 60.
        margin : bool, optional
            If this is a float > 0, all tracks within a bounding box obtained by adding up this value to the point's
            bounding box are considered. The default is None.
        interpolate : bool, optional
            If True, all tracks within the distance of a shp.LineString built out of the single points will be
            considered. This avoids that connecting tracks are not contained in the subgraph, e.g., if they are short
            and the vehicle moves fast. The default is True.
        cutoff : int, optional
            Maximal length of a railway path. Avoids large computation times in large networks. The default is None.
        distance_start_end : float, optional
            Distance to start and end points in which potential start and end tracks are looked for. The default is 15.

        Returns
        -------
        valid_paths_list : list(list(tuple))
            A list of railway paths that can be used to create a railpos.railwaypaths.RailwayPath object. One path
            element consists of a list of (track, direction)-tuples. Directions in the output are represented by (1,-1).

        """

        #  It must be avoided that required nodes are not in graph. This means that the distance for selecting start
        #  and end tracks should not be smaller than the allowed distance to points for tracks in the subgraph
        #  Alternatively use distance_start_end = min(distance_start_end, distance)
        distance = max(distance, distance_start_end)
        subgraph = self.subgraph_around_data(data, distance=distance, margin=margin, interpolate=interpolate)
        try:
            start = data.geometry.iloc[0]
            end = data.geometry.iloc[-1]
        except IndexError:
            start = data[0]
            end = data[-1]

        start_tracks = self.tracks_near_data(start, distance_start_end)
        end_tracks = self.tracks_near_data(end, distance_start_end)

        valid_paths_list = list()

        for start_track in start_tracks:
            for end_track in end_tracks:
                start_direction = None
                if start_track == end_track:
                    try:
                        distances = self.geo_df.iloc[start_track].geometry.project(data.geometry)
                        start_direction = np.sign(distances.iloc[-1] - distances.iloc[0])
                    except:
                        distances = self.geo_df.iloc[start_track].geometry.project(data)
                        start_direction = np.sign(distances[-1] - distances[0])
                    if start_direction == 0:
                        start_direction = np.sign(np.mean(np.diff(distances)))
                if start_direction == 0:
                    continue
                valid_paths_temp = valid_paths(g=subgraph, start_track=start_track, end_track=end_track,
                                               start_direction=start_direction, cutoff=cutoff)
                valid_paths_list.extend(valid_paths_temp)

        return valid_paths_list

    def to_geopackage(self, filename: Union[str, os.PathLike], cols_to_convert=None):
        """


        Parameters
        ----------
        filename : str or path-like
            File path for saving the map data.
        cols_to_convert : list, optional
            List of column names that will be converted to a string. If None, will be set to
            ["connected_start", "connected_end"]. The default is None.

        Returns
        -------
        None.

        """
        if cols_to_convert is None:
            cols_to_convert = ["connected_start", "connected_end"]
        tmp = self.geo_df.copy()
        for col in cols_to_convert:
            tmp.loc[:, col] = [repr(val) for val in tmp[col]]
        tmp.to_file(filename, driver="GPKG")


def valid_paths(g: nx.DiGraph, start_track: int, end_track: int, start_direction=None, cutoff=None):
    """
    Get all valid (railway) paths from a start track to an end track given a graph representing valid connections.

    Parameters
    ----------
    g : nx.DiGraph
        Contains connections as described in create_graph_directed_connections in RailwayMap.
    start_track : int
        The start track of the railway path.
    end_track : int
        The end track of the railway path.
    start_direction : bool or int(+-1), optional
        If this is specified, the orientation of the first track in the path is fixed. The default is None.
    cutoff : int, optional
        Maximal length of a railway path. Avoids large computation times in large networks. The default is None.

    Returns
    -------
    list_valid_paths : list(list(tuple))
            A list of railway paths that can be used to create a railpos.railwaypaths.RailwayPath object. One path
            element consists of a list of (track, direction)-tuples. Directions in the output are represented by (1,-1).

    """

    list_valid_paths = list()

    if start_direction is not None:

        if nx.has_path(g, (start_track, start_direction), (end_track, 1)):
            list_valid_paths.extend(list(nx.all_simple_paths(g, (start_track, start_direction),
                                                             (end_track, 1), cutoff=cutoff)))
        if nx.has_path(g, (start_track, start_direction), (end_track, -1)):
            list_valid_paths.extend(list(nx.all_simple_paths(g, (start_track, start_direction),
                                                             (end_track, -1), cutoff=cutoff)))
    else:
        if nx.has_path(g, (start_track, 1), (end_track, 1)):
            list_valid_paths.extend(list(nx.all_simple_paths(g, (start_track, 1),
                                                             (end_track, 1), cutoff=cutoff)))
        if nx.has_path(g, (start_track, -1), (end_track, 1)):
            list_valid_paths.extend(list(nx.all_simple_paths(g, (start_track, -1),
                                                             (end_track, 1), cutoff=cutoff)))
        if nx.has_path(g, (start_track, 1), (end_track, -1)):
            list_valid_paths.extend(list(nx.all_simple_paths(g, (start_track, 1),
                                                             (end_track, -1), cutoff=cutoff)))
        if nx.has_path(g, (start_track, -1), (end_track, -1)):
            list_valid_paths.extend(list(nx.all_simple_paths(g, (start_track, -1),
                                                             (end_track, -1), cutoff=cutoff)))

    # include single-track paths

    if start_track == end_track:
        if start_direction is None:
            list_valid_paths.append([(start_track, 1)])
            list_valid_paths.append([(start_track, -1)])
        else:
            list_valid_paths.append([(start_track, start_direction)])

    return list_valid_paths


if __name__ == '__main__':
    pass

