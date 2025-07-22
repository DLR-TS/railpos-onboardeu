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
Contains helper functions to handle timestamps of sensor data.
-------
"""

# Common Libs (numpy, pandas, ...)
import numpy as np


def create_kf_timestamps(time_arrays: dict, dt_kf: float, timespan: str):
    """
    Create equally spaced timestamp vector for a Kalman filter.
    Covers the time span of the time arrays. Covers either the intersection
    or the maximal time span of the individual time arrays.

    Parameters
    ----------
    time_arrays : dict
        Dictionary containing all time arrays of sensor data to be considered (covered) in the output time vector.
        Should all be in unix format. All must have the same units.
    dt_kf : float
        Time difference of subsequent timestamps of the Kalman filter. Must have the same unit as the time arrays.
    timespan : str
        'intersect' or 'max'. If 'intersect', output timestamps cover the intersecting time span of all sensor time
         arrays. If 'max', it covers the maximal time span contained in any of the time arrays.

    Returns
    -------
    t_kf : np.ndarray (shape = (n,))
        Equally spaced timestamps to be used in a Kalman filter.

    """

    if timespan == 'intersect':
        t_start = max([np.nanmin(time_arrays[key]) for key in time_arrays])
        t_start = np.ceil(t_start / dt_kf) * dt_kf
    elif timespan == 'max':
        t_start = min([np.nanmin(time_arrays[key]) for key in time_arrays])
        t_start = np.floor(t_start / dt_kf) * dt_kf
    if str(dt_kf).find('.') == -1:
        round_nr = 0
    else:
        round_nr = len(str(dt_kf)[str(dt_kf).find('.') + 1:])
    t_start = np.round(t_start, round_nr)
    if timespan == 'intersect':
        t_end = min([np.nanmax(time_arrays[key]) for key in time_arrays])
        t_end = np.floor(t_end / dt_kf) * dt_kf
    elif timespan == 'max':
        t_end = max([np.nanmax(time_arrays[key]) for key in time_arrays])
        t_end = np.ceil(t_end / dt_kf) * dt_kf
    t_end = np.round(t_end, round_nr)

    n = int((t_end - t_start) / dt_kf) + 1
    t_kf = np.linspace(0, t_end - t_start, n)
    # Prospective linspace solution to solve problems with long time vectors
    t_kf += t_start
    return t_kf


def assign_meastime_to_reftime(time_ref: np.ndarray, time_meas: np.ndarray, causal_flag: bool = True):
    """
    Assign timestamps of sensor measurements to a reference time array (e.g., timestamps of a Kalman filter).

    Parameters
    ----------
    time_ref : np.ndarray
        Reference time array. Same unit as time_meas.
    time_meas : np.ndarray
        Time array of the measurement data. Same unit as time_ref.
    causal_flag : bool, optional
        If True, a measurement timestamp is always assigned to a future reference timestamp. The default is True.

    Returns
    -------
    measind_ref : np.ndarray(int)
        Contains for each timestamp in the reference time array an index referring to the timestamp array of the sensor
        measurements. More precisely, if measind_ref[k] == -1, there is no measurement timestamp assigned to
        time_ref[k]. If measind_ref[k] == i >=0, then time_meas[i] is assigned to time_ref[k].

    """
    n_steps = len(time_ref)
    n_meas = len(time_meas)

    # initialize index vector
    measind_ref = -np.ones(n_steps, dtype='int64')

    # threshold times
    if causal_flag:
        time_thr = time_ref
    else:
        time_thr = time_ref[:-1] + np.diff(time_ref) / 2
        time_thr = np.append(time_thr,
                             np.maximum(time_ref.max(), time_meas.max()))

    k = 0  # index for the reference times

    for i in range(n_meas):  # index for the measurement times

        # find the corresponding k
        while time_thr[k] < time_meas[i] and k < n_steps - 1:
            k += 1

        if causal_flag:
            measind_ref[k] = i
            # make sure the final assignment is causal
            if k == n_steps - 1:
                if time_thr[k] < time_meas[i]:
                    measind_ref[k] = -1
        else:
            # first assignment
            if measind_ref[k] == -1:
                measind_ref[k] = i
                # compute deviation
                dev_best = abs(time_ref[k] - time_meas[i])
            else:
                # check if there is a better i for k
                dev_new = abs(time_ref[k] - time_meas[i])
                if dev_new < dev_best:
                    measind_ref[k] = i
                    dev_best = dev_new

    return measind_ref


def time_association(time_target: np.ndarray, time_input: np.ndarray, causal: bool = True):
    """
    Associate the entries of an input time vector with the entries of
    an equally sampled target time vector.


    Parameters
    ----------
    time_target : np.ndarray
        Vector with n_target samples, sorted and equally spaced (e.g. timestamps of a Kalman filter).
    time_input : np.ndarray
        Vector with n_input samples (e.g., of sensor data).
    causal : bool, optional
        If True, a measurement timestamp is always assigned to a future reference timestamp. The default is True.

    Returns
    -------
    ind_input : np.ndarray
        Vector with n_input entries.
    time_associated : np.ndarray
        DESCRIPTION.

    """

    dt_target = np.mean(np.diff(time_target))
    ind_input = (time_input - time_target[0]) / dt_target
    if causal:
        ind_input = np.ceil(ind_input)
    else:
        ind_input = np.round(ind_input)

    ind_input = np.maximum(ind_input, 0)
    ind_input = np.minimum(ind_input, time_target.size - 1)
    ind_input = ind_input.astype('int')

    time_associated = time_target[ind_input]

    return ind_input, time_associated


if __name__ == '__main__':
    pass
