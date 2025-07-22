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
Provides a class and functionalities to handle railway paths (paths in a railway network), including path finding for
offline positioning and path handling for online positioning.
-------
"""

import copy
import numpy as np
import geopandas as gpd
import shapely as shp
from typing import List, Tuple, Iterable
from numpy.typing import ArrayLike

from railpos.helpers import is_sublist
from railpos.railwaymaps import RailwayMap


class RailwayPath:
    """
    Class for representing railway paths.

    A railway path is a sequence of connected railway tracks in a
    railway network that can be visited by a railway vehicle in one
    motion without intermediate changes of direction.

    Attributes:
        railway_map (RailwayMap): An object that describes the railway map.
        tracks (List[int]): The N tracks in a railway path. Integer indices of the rows in railway_map.geo_df.
        aligned (Union[List[bool], List[int]]): The N orientations of tracks with respect to the railway path.
            A track is aligned with the path if an increase in on-path distance corresponds to an increase in
            on-track distance.
        lengths (List[float]): The N lengths of the tracks.
        cumlengths (np.ndarray): The cumulative sum of lengths.
        offset (float): An offset distance used to model arbitrary vehicle-driven distances.
            Vehicle distance = on-path distance + offset distance.
        geometry (shp.MultiLineString or shp.LineString): Path geometry.


    """

    def __init__(self, railway_path: List[Tuple], railway_map: RailwayMap):
        """
        Initialize a railway path.

        Parameters
        ----------
        railway_path : List[Tuple]
            A list of 2-tuples that represents a path in the railway network.
            Each tuple comprises two values. First, an index integer of the track in railway_map.geo_df.
            Second, a value (1 or -1) that describes the orientation of the track in the railway path.
            Instead of 1 and -1, True and False are also accepted as orientations (1 corresponds to True).
        railway_map : RailwayMap
            A RailwayMap object which contains the geometries of the railway network. The path refers to this network.
            This means that the tracks refer to indices of the GeoDataFrame saved in RailwayMap.

        Returns
        -------
        None.

        """

        self.railway_map = railway_map
        self.tracks = [item[0] for item in railway_path]
        self.aligned = [item[1] > 0 for item in railway_path]
        self.lengths = [geom.length for geom in railway_map.geo_df.geometry.iloc[self.tracks]]
        self.cumlengths = np.cumsum(self.lengths)
        self.offset = 0

    def recompute_lengths(self):
        """
        Re-computes the lengths and cumlengths attributes if a path is updated.

        Returns
        -------
        None.

        """
        self.lengths = [geom.length for geom in self.railway_map.geo_df.geometry.iloc[self.tracks]]
        self.cumlengths = np.cumsum(self.lengths)

    @property
    def geometry(self):
        """
        Create a the geometry of the path.

        Returns
        -------
        path_ls : Union[shp.MultiLineString, shp.LineString]
            A (Multi)LineString representing the geometry of the given path.

        """
        path_ls = shp.unary_union([self.railway_map.geo_df.geometry.iloc[track] if isaligned else self.railway_map.geo_df.geometry.iloc[
            track].reverse() for (track, isaligned) in zip(self.tracks, self.aligned)])
        return path_ls

    def reverse(self):
        """
        Flip the path - only for offline application.

        Returns
        -------
        None.

        """
        self.tracks.reverse()
        self.aligned.reverse()
        self.aligned = [not al for al in self.aligned]

        self.recompute_lengths()

    @property
    def nr_tracks_in_path(self):
        """
        Compute the number of tracks in the current railway path

        Returns
        -------
        int
            Number of tracks in the paths.

        """
        return len(self.tracks)

    @property
    def rpath_length(self):
        """
        Compute the length of the current railway path.

        Returns
        -------
        float
            Length of the current railway path.

        """
        return self.cumlengths[-1]

    def get_candidates(self, at_end: bool = True):
        """
        Return candidate tracks for railway path extension.

        Parameters
        ----------
        at_end : bool, optional
            If True, adjacent tracks of the last track are returned.
            If False, adjacent tracks of the first track are returned. The default is True.

        Returns
        -------
        cands : List[int]
            Integer indices of candidates for path extension.

        """
        if at_end:
            if self.aligned[-1]:
                cands = self.railway_map.geo_df.connected_end.iloc[self.tracks[-1]]
            else:
                cands = self.railway_map.geo_df.connected_start.iloc[self.tracks[-1]]
        else:
            if self.aligned[0]:
                cands = self.railway_map.geo_df.connected_start.iloc[self.tracks[0]]
            else:
                cands = self.railway_map.geo_df.connected_end.iloc[self.tracks[0]]
        return cands

    def get_candidate_alignment(self, cand: int, at_end: bool = True):
        """
        Return alignment of a candidate for railway path extension.

        Parameters
        ----------
        cand : int
            Integer index of a track candidate for a path extension.
        at_end : bool, optional
            If True, adjacent tracks of the last track are returned.
            If False, adjacent tracks of the first track are returned.
            The default is True.

        Returns
        -------
        is_aligned : bool

        """
        if at_end:
            if self.tracks[-1] in self.railway_map.geo_df.connected_start.iloc[cand]:
                is_aligned = True
            else:
                is_aligned = False

        else:
            if self.tracks[0] in self.railway_map.geo_df.connected_end.iloc[cand]:
                is_aligned = True
            else:
                is_aligned = False

        return is_aligned

    def extend_single(self, at_end: bool = True):
        """
        Appends one track to the current railway path and creates new railway paths
        if there is more than one adjacent track.

        Parameters
        ----------
        at_end : bool, optional
            If True, tracks are appended at the end of the railway path.
            If False, tracks are appended before the first track of the railway path.
            The default is True.

        Returns
        -------
        List[RailwayPath]:
            A list of newly created railway paths.

        """
        # find candidates
        cands = self.get_candidates(at_end)
        n_cands = len(cands)

        # store the offset
        offset_temp = 1.0 * self.offset

        # no candidates, no further actions
        if cands == []:
            return []

        candalignments = []
        for i, cand in enumerate(cands):
            candalignments.append(self.get_candidate_alignment(cand, at_end))

        # append / insert dummy variables before iteration
        if at_end:
            self.aligned.append(True)
            self.tracks.append(0)
            self.lengths.append(0.0)

        else:
            self.aligned.insert(0, True)
            self.tracks.insert(0, 0)
            self.lengths.insert(0, 0.0)

        # list for the new path objects
        newpaths = []

        # iterate over candidates
        for i, cand in enumerate(cands):
            if at_end:
                self.aligned[-1] = candalignments[i]
                self.tracks[-1] = cand
                self.lengths[-1] = self.railway_map.geo_df.geometry.iloc[cand].length
                self.offset = offset_temp

            else:
                self.aligned[0] = candalignments[i]
                self.tracks[0] = cand
                self.lengths[0] = self.railway_map.geo_df.geometry.iloc[cand].length
                self.offset = offset_temp - self.lengths[0]

            # simple but not most efficient re-computation of the cumlengths
            self.cumlengths = np.cumsum(self.lengths)

            # for n candidate tracks a list with (n-1) new paths is created
            if i < n_cands - 1:
                newpaths.append(self.copy())

        return newpaths

    def remove_single(self, at_end: bool = True):
        """
        Removes a single track from the current railway path.

        Parameters
        ----------
        at_end : bool, optional
            If True, tracks are removed from the end of the railway path.
            If False, the first track is removed from the railway path. The default is True.

        Returns
        -------
        None.

        """
        # do nothing if there is only one track
        if self.nr_tracks_in_path == 1:
            pass

        if at_end:
            self.aligned.pop()
            self.tracks.pop()
            self.lengths.pop()
        else:
            self.aligned.pop(0)
            self.tracks.pop(0)
            offset_update = self.lengths.pop(0)
            self.offset += offset_update
        self.cumlengths = np.cumsum(self.lengths)

    def d_on_path(self, distance_vehicle: float):
        """


        Parameters
        ----------
        distance_vehicle : float
            A vehicle distance.

        Returns
        -------
        float
            An on-path distance (computed by subtracting self.offset).

        """
        return distance_vehicle - self.offset

    def extend_from_margin(self, distance: float, margin: float):
        """
        Appends tracks to the current railway path such that a provided (distance + margin) is on the extended railway
        path.
        Creates new railway paths if several options are possible.

        Parameters
        ----------
        distance : float
            Vehicle distance.
        margin : float
            Margin for extension so that distance + margin is on the path.

        Returns
        -------
        newpaths : List[RailwayPath]
            A list of newly created railway paths.

        """
        # - extends the path given a distance
        # - returns a list of paths and updates the current path
        # - path extension not possible - number of tracks stays constant
        # - path extension possible - must update the offset, tracks
        # - recursive implementation. function is called for the newly created paths

        # positive margin - extend at the end of the path
        # negative margin - extend at the start
        if margin < 0:
            at_end = False
            extension_needed = (self.d_on_path(distance + margin) < 0)
        else:
            at_end = True
            extension_needed = (self.d_on_path(distance + margin) >= self.rpath_length)

        newpaths = []

        while extension_needed:
            n_before = self.nr_tracks_in_path

            newpaths.extend(self.extend_single(at_end=at_end))

            if margin < 0:
                extension_needed = (self.d_on_path(distance + margin) < 0)
            else:
                extension_needed = (self.d_on_path(distance + margin) >= self.rpath_length)

            if self.nr_tracks_in_path == n_before:
                extension_needed = False

        for rpath in newpaths:
            newpaths.extend(rpath.extend_from_margin(distance, margin))

        return newpaths

    def copy(self):
        """
        Generates a copy of the RailwayPath object.
        Note that the railwaymap is not copied.
        Both the copy and the original refer to the same railwaymap object.

        Returns
        -------
        cpy : RailwayPath
            Copy of self.

        """
        cpy = copy.copy(self)  # create a shallow copy

        # create copies of the entries
        cpy.tracks = cpy.tracks.copy()
        cpy.aligned = cpy.aligned.copy()
        cpy.lengths = cpy.lengths.copy()
        cpy.offset = 1.0 * cpy.offset

        cpy.cumlengths = cpy.cumlengths.copy()

        return cpy

    def path2track(self, dveh: float):
        """
        Converts vehicle distances to on-track distances.

        Parameters
        ----------
        dveh : float
            A vehicle distance.

        Returns
        -------
        d_track : float
            The on-track distance.
        ind_track : int
            Integer index of the track.

        """

        d_path = self.d_on_path(dveh)

        # check for valid d_path and clip
        d_path = np.clip(d_path, 0, self.cumlengths[-1])

        # compute the track index in the path
        i_in_path = np.sum(d_path > self.cumlengths)

        # what is the track index in the overall track list?
        ind_track = self.tracks[i_in_path]

        # compute on-track distance
        if i_in_path > 0:
            d_track = d_path - self.cumlengths[i_in_path - 1]
        else:
            d_track = d_path
        # adjust according to track orientation in path
        if not self.aligned[i_in_path]:
            d_track = self.lengths[i_in_path] - d_track

        return d_track, ind_track

    def path2xy(self, dveh: float):
        """
        Converts vehicle distances to points.

        Parameters
        ----------
        dveh : float
            A vehicle distance.

        Returns
        -------
        pt : shp.geometry.Point
            Global position in the CRS of self.rmap.geo_df.

        """
        dtrack, itrack = self.path2track(dveh)

        pt = self.railway_map.geo_df.geometry.iloc[itrack].interpolate(dtrack)

        return pt

    def trim_from_margin(self, distance: float, margin: float):
        """
        Shorten the path given a distance and a margin such that the first or last track in the path contains the point
        distance+margin.
        Only one-sided removal of tracks from the path.

        Behavior for 0: shortens only at end

        Parameters
        ----------
        distance : float
            A vehicle distance.

        margin : float
            A margin to be covered by the path starting from the vehicle distance.


        Returns
        -------
        None.

        """
        # do nothing if only one track
        if self.nr_tracks_in_path == 1:
            pass

        # positive margin - shorten at the end of the path
        # negative margin - shorten at the start
        if margin < 0:
            at_end = False
            removal_needed = (self.d_on_path(distance + margin) >= self.lengths[0])
        else:
            at_end = True
            removal_needed = (self.d_on_path(distance + margin) < self.rpath_length - self.lengths[-1])  # corrected

        while removal_needed and self.nr_tracks_in_path > 1:

            self.remove_single(at_end=at_end)

            if margin < 0:
                removal_needed = (self.d_on_path(distance + margin) >= self.lengths[0])
            else:
                removal_needed = (self.d_on_path(distance + margin) < self.rpath_length - self.lengths[-1])  # corrected


    def points2path(self, points: Iterable[shp.geometry.Point]):
        """
        Project points onto the path.

        Parameters
        ----------
        points : Iterable[shp.geometry.Point]
            An iterable (list, tuple, GeoSeries) of points.
            The points will be projected onto the path by finding points on the path that are closest.
            The points and the map must be resolved in a common Cartesian CRS.
            That is, position must be in meters and not degrees.

        Returns
        -------
        projected_points : Iterable[shp.geometry.Point]
            The projected points on the path.
        projection_errors : Iterable[float]
            Projection errors.
        projection_distances : Iterable[float]
            The distances (including offset) on the path.

        """
        projection_errors = self.geometry.distance(points)
        projection_distances = self.geometry.project(points)  # distance on geometry
        projected_points = self.geometry.interpolate(projection_distances)
        projection_distances += self.offset  # add offset to get vehicle distance
        return projected_points, projection_errors, projection_distances


def rpaths_from_gnss(rmap: RailwayMap, gdf_gnss: gpd.GeoDataFrame, thr_meter: float = 25, cutoff: int = 50,
                     distance_start_end: float = 15, **kwargs):
    """
    Get RailwayPath objects for all valid paths for GNSS data. The paths are given by all valid paths in the railway
    network that
    connect all tracks that are within a distance to the first GNSS data point to all tracks that are within a distance
    to the last data point.

    Parameters
    ----------
    rmap : RailwayMap
        The RailwayMap object containing the railway network geometries.
    gdf_gnss : gpd.GeoDataFrame
        A GeoDataFrame with GNSS positions as geometries.
    thr_meter : float, optional
        Tracks within this distance to the geometries in gdf_gnss are considered to be included in the paths.
        The default is 25.
    cutoff : int, optional
        Maximal path length. To avoid computational overload. The default is 50.
    distance_start_end : float, optional
        Tracks within this distance to the start and end geometries are considered as start and end tracks.
        The default is 15.
    **kwargs :
        Keyword arguments to hand over to valid_paths_for_data_points of RailwayMap.

    Returns
    -------
    rpaths : List[RailwayPath]
        RailwayPath objects for all valid paths for GNSS data.

    """
    rpaths = rmap.valid_paths_for_data_points(gdf_gnss, distance=thr_meter, cutoff=cutoff,
                                              distance_start_end=distance_start_end, **kwargs)
    rpaths = [RailwayPath(rpath, rmap) for rpath in rpaths]

    return rpaths


def rpath_projection_errors(rpaths: List[RailwayPath], gdf_gnss: gpd.GeoDataFrame, w_mean: float = 1,
                            w_median: float = 1, w_ntracks: float = 0.1):
    """
    Calculate projection errors of GNSS data to a list of RailwayPath objects and the best path by optionally weighted
    performance measures.

    Parameters
    ----------
    rpaths : List[RailwayPath]
        List of RailwayPath objects.
    gdf_gnss : gpd.GeoDataFrame
        Contains the GNSS data as geometries.
    w_mean : float, optional
        Multiple to weight the mean projection error in the performance measure. The default is 1.
    w_median : float, optional
        Multiple to weight the median projection error in the performance measure. The default is 1.
    w_ntracks : float, optional
        Multiple to weight the number of tracks in the path in the performance measure. The default is 0.1.

    Returns
    -------
    projection_errors_matrix : np.ndarray (shape: (number of paths, number of GNSS data points))
        Contains all projection errors of the GNSS points to all paths.
    i_best : int
        Index referring to the best performing path in the list with respect to the weighted sum of the different
        performance measures.

    """
    n_paths = len(rpaths)
    n_gnss = len(gdf_gnss)
    projection_errors_matrix = np.zeros((n_paths, n_gnss))
    for i, rpath in enumerate(rpaths):
        _, projection_errors, _ = rpath.points2path(gdf_gnss.geometry)
        projection_errors_matrix[i] = projection_errors
    perf0 = projection_errors_matrix.mean(axis=1)
    perf1 = np.median(projection_errors_matrix, axis=1)
    perf2 = np.array([rpath.nr_tracks_in_path for rpath in rpaths])
    i_best = np.argmin(w_mean * perf0 + w_median * perf1 + w_ntracks * perf2)
    return projection_errors_matrix, i_best


def rpath_coverage(rpaths: List[RailwayPath], gdf_gnss: gpd.GeoDataFrame):
    """
    Compute the covered on-path distance by projecting GNSS points onto paths.

    Parameters
    ----------
    rpaths : List[RailwayPath]
        A list of RailwayPath objects.
    gdf_gnss : gpd.GeoDataFrame
        Contains GNSS data.

    Returns
    -------
    covered_distance : np.ndarray
        Contains the covered on-path distance per path.

    """
    n_paths = len(rpaths)
    covered_distance = np.zeros(n_paths)
    for i, rpath in enumerate(rpaths):
        _, _, projection_distances = rpath.points2path(gdf_gnss.geometry)
        covered_distance[i] = np.abs(np.max(projection_distances) - np.min(projection_distances))
    return covered_distance


def rpath_check_forbidden(rpaths: List[RailwayPath], forbidden_connections: List[list]):
    """
    Check a list of RailwayPath objects for forbidden connections in the paths.

    Parameters
    ----------
    rpaths : List[RailwayPath]
        A list of RailwayPath objects.
    forbidden_connections : List[list] (?)
    # TODO check type hint and add description
        DESCRIPTION.

    Returns
    -------
    forbidden_flag : np.ndarray(bool)
        Flag at index i is True if there was a forbidden connection in path i.

    """

    forbidden_flag = np.zeros(len(rpaths), dtype=bool)

    for i, rpath in enumerate(rpaths):

        for forb in forbidden_connections:
            if is_sublist(forb, rpath.tracks):
                forbidden_flag[i] = True
                break

    return forbidden_flag


def flag_movement_direction_not_into_path_direction(rpath: RailwayPath, points: ArrayLike):
    """
    Checks if the path distances of projections of GNSS data to a RailwayPath are increasing (with order/time).
    This is necessary for offline positioning only.

    Parameters
    ----------
    rpath : RailwayPath
        A RailwayPath object.
    points : ArrayLike[shp.Point]
        The positions that will be projected to the path.

    Returns
    -------
    flag : bool
        If False, the on-path distances are increasing (in mean) with increasing order.

    """
    distances = rpath.geometry.project(points)
    flag = (np.nanmean(np.sign(np.diff(distances))) < 0)
    return flag


def redundant_paths(rpaths: List[RailwayPath], ignore_offset: bool = True):
    """
    Find redundant paths in a list of paths.

    Parameters
    ----------
    rpaths : List[RailwayPath]
        A list of RailwayPath objects.
    ignore_offset : bool, optional
        If True, the offset attribute of the RailwayPath objects is ignored when equality is checked.
        The default is True.

    Returns
    -------
    redundant_flag : np.ndarray
        Contains flags for redundancy. If the flag at index i is True, then there is already a RailwayPath in the list
        at an index j < i which is identical (same tracks, same alignments, optionally same offset) as the path at
        index i.

    """
    n_paths = len(rpaths)
    redundant_flag = np.zeros(n_paths, dtype='bool')
    for i in range(n_paths):
        if not redundant_flag[i]:
            for j in range(i + 1, n_paths):
                same_tracks = (rpaths[i].tracks == rpaths[j].tracks)
                same_alignment = rpaths[i].aligned == rpaths[j].aligned
                redundant_condition = same_tracks and same_alignment
                if not ignore_offset:
                    same_offset = abs(rpaths[i].offset - rpaths[j].offset) < 0.001
                    redundant_condition = redundant_condition and same_offset
                if redundant_condition:
                    redundant_flag[j] = True

    return redundant_flag


if __name__ == '__main__':
    pass
