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
Helper functions for data pre-processing and positioning.
"""

# Common Libs (numpy, pandas, ...)
import numpy as np
import scipy
import shapely as shp
from scipy import interpolate
from railpos.timestamps import assign_meastime_to_reftime


def integrate_signal(signal: np.ndarray, timestamps: np.ndarray, bias: float = None):
    """
    Integrate a signal with respect to time.

    Parameters
    ----------
    signal : np.ndarray(float)
        Signal which will be integrated.
    timestamps : np.ndarray(float)
        Timestamps of the signal.
    bias : float, optional
        Will be subtracted from signal. If None, the mean of the signal is used.
        The default is None.

    Returns
    -------
    signal_integrated : np.ndarray
        Integrated signal.

    """
    if bias is None:
        bias = np.mean(signal)

    signal_integrated = signal.copy() - bias
    signal_integrated[1:] = np.cumsum(np.diff(timestamps) * (signal[1:] - bias))

    return signal_integrated


def detect_duplicates(values: np.ndarray, ignore_zero_values: bool = True):
    """
    Detect positions in an array where the value is identical to the value in the precedent position.

    Parameters
    ----------
    values : np.ndarray.
        Input array to check for duplicates.
    ignore_zero_values : bool, optional
        If True, the value zero is ignored in the duplicate detection.
        The default is True.

    Returns
    -------
    duplicate_flag : np.ndarray(bool)
        Is True in index i if values[i] is a duplicate of values[i-1].

    """
    duplicate_flag = np.zeros(len(values), dtype='bool')
    for i, val in enumerate(values):
        if i == 0:
            pass

        if val == values[i - 1]:
            duplicate_flag[i] = True

    if ignore_zero_values:
        duplicate_flag[values == 0] = False

    return duplicate_flag


def angle(v1: np.ndarray, v2: np.ndarray):
    """
    Compute the angle between two vectors v1 and v2.

    Parameters
    ----------
    v1 : np.ndarray
        One of the input arrays for the angle computation.
    v2 : np.ndarray
        Second input array for the angle computation.

    Returns
    -------
    Float
        Angle between v1 and v2 (degrees).

    """
    scalar_product = np.inner(v1, v2)
    normalized = scalar_product / np.sqrt(np.inner(v1, v1) * np.inner(v2, v2))
    return np.arccos(normalized) * 180 / np.pi


def filtfilt_lp(y: np.ndarray, f_cut: float = 0.05, filter_order=2, f_nyq=50):
    """
    Low pass filter an input signal using forward and backward filtering (scipy.signal.filtfilt).

    Parameters
    ----------
    y : np.ndarray
        Array with signal values.
    f_cut : float, optional
        Frequency used for low pass filtering. The default is 0.05.
    filter_order : int, optional
        Order of the utilised (Butterworth) filter. The default is 2.
    f_nyq : float, optional
        Nyquist frequency of the signal. The default is 50.

    Returns
    -------
    y_filt : np.ndarray
        Filtered values of y.

    """
    b_filt, a_filt = scipy.signal.butter(filter_order, f_cut / f_nyq)
    y_filt = scipy.signal.filtfilt(b_filt, a_filt, y)
    return y_filt


def moving_average(x: np.ndarray, w: int):
    """
    Moving average filtering of a signal x with window length w

    Parameters
    ----------
    x : np.ndarray
        Input array.
    w : int
        Window length (number of samples).

    Returns
    -------
    np.ndarray
        Filtered input array.

    """
    return np.convolve(x, np.ones(w), 'same') / w


def point2array(pt: shp.Point):
    """
    Convert a shp.Point to a coordinate numpy array.

    Parameters
    ----------
    pt : shp.Point
        The point whose coordinates are converted to a np.ndarray.

    Returns
    -------
    np.ndarray
        Array containing the coordinates of the point pt.

    """
    return np.array(pt.coords[0])


def is_sublist(list1: list, list2: list):
    """
    Check whether list1 is contained in list2

    Parameters
    ----------
    list1 : list
        First list for which it is checked if it is contained in the second list.
    list2 : list
        Second list for which it is checked if it contains the first list.

    Returns
    -------
    bool
        True if list1 is contained in list2.

    """
    for i in range(len(list2) - len(list1) + 1):
        if list2[i:i + len(list1)] == list1:
            return True
    return False


def driving_direction(imu_acc: np.ndarray, imu_times: np.ndarray, gnss_speed: np.ndarray, gnss_times: np.ndarray,
                      bias=None):
    """
    Classify forward/backward motion from IMU and GNSS value arrays.

    Parameters
    ----------
    imu_acc : np.ndarray(float)
        Contains longitudinal acceleration measured by the IMU.
    imu_times : np.ndarray(float)
        Timestamps of the acceleration values.
    gnss_speed : np.ndarray(float)
        GNSS speed values.
    gnss_times : np.ndarray(float)
        Timestamps of the GNSS speed values.
    bias : float, optional
        Bias of the acceleration values. If None, the mean is calculated.
        The default is None.

    Returns
    -------
    direction : int (+-1)
        Driving direction with respect to the orientation of the acceleration (IMU or vehicle coordinate frame).
        1 is forward, -1 is backward motion with respect to this coordinate frame.
    speed_errors : np.ndarray
        Difference between integrated longitudinal acceleration * calculated direction and GNSS speed.

    """

    # dead-reckoning of the IMU data
    speed_imu = integrate_signal(imu_acc, imu_times, bias=bias)

    # index vector, as long as the IMU data
    indices = assign_meastime_to_reftime(imu_times, gnss_times)

    # picks a subset of GNSS data
    speed_gnss_selection = gnss_speed[indices[indices >= 0]]

    # picks a subset of the integrated IMU data
    speed_imu_selection = speed_imu[indices >= 0]

    speed_errors = []
    for orientation in [-1, 1]:
        speed_difference = orientation * speed_gnss_selection - speed_imu_selection
        speed_errors.append(np.mean(np.abs(speed_difference)))

    if speed_errors[0] < speed_errors[1]:
        direction = -1  # backwards
    else:
        direction = 1  # forwards

    return direction, speed_errors


def standstill_from_gnss(gnss_times: np.ndarray, gnss_speed: np.ndarray, output_time: np.ndarray,
                         threshold: float = 0.1):
    """
    Calculate standstill flags based on GNSS speed and provide (interpolate) them for an output time vector (e.g.,
    the timestamps of a Kalman filter). It is assumed that the sampling frequency of the output time is higher than
    the sampling frequency of the GNSS data.

    Parameters
    ----------
    gnss_times : np.ndarray(float)
        GNSS timestamps.
    gnss_speed : : np.ndarray(float)
        GNSS speed.
    output_time: np.ndarray(float)
        Timestamps for which the standstill flag is returned.
    threshold: float, optional
        GNSS speeds below this threshold will be regarded as standstill. The default is 0.1.

    Returns
    -------
    standstill_kf : np.ndarray(bool)
        Contains the standstill flags with respect to the timestamps in output_time.

    """
    standstill_gnss = (gnss_speed < threshold)
    interpol_left = interpolate.interp1d(gnss_times, standstill_gnss, kind='previous',
                                         bounds_error=False, fill_value='extrapolate')
    standstill_kf_left = interpol_left(output_time)
    standstill_kf_left = np.array([bool(i) for i in standstill_kf_left])

    interpol_right = interpolate.interp1d(gnss_times, standstill_gnss, kind='next',
                                          bounds_error=False, fill_value='extrapolate')
    standstill_kf_right = interpol_right(output_time)
    standstill_kf_right = np.array([bool(i) for i in standstill_kf_right])

    standstill_kf = np.logical_and(standstill_kf_left, standstill_kf_right)
    standstill_shift = np.concatenate([np.zeros(1, dtype=bool),
                                       np.logical_and(standstill_kf_left[1:], standstill_kf_right[:-1])])
    standstill_kf = np.logical_or(standstill_kf, standstill_shift)
    return standstill_kf


if __name__ == "__main__":
    pass
