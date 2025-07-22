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
Provides functions and classes for Kalman filtering and related tasks.
-------
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from railpos.helpers import point2array
from railpos.timestamps import time_association
from railpos.railwaypaths import RailwayPath


def kf_gain(m_mat: np.ndarray, s_mat: np.ndarray):
    """
    Computes the Kalman gain given.
    The equation is $K = M^T S^{-1}$.
    For the linear case $M = P H^T$ and $S = H P H^T + R$.
    Instead of computing an inverse, a linear systems of equations is solved.

    Parameters
    ----------
    m_mat : np.ndarray
        The n-by-m cross-covariance of the predicted state and measurement.
    s_mat : np.ndarray
        The m-by-m covariance of the predicted measurement (also the residual covariance).

    Returns
    -------
    k_mat : np.ndarray
         Kalman gain.

    """

    k_mat = np.linalg.solve(s_mat, m_mat.T).T
    return k_mat


def kf_meas_scalar(xh: np.ndarray, p_mat: np.ndarray, h_vec: np.ndarray, r: float, y: float):
    """
    Performs a linear KF measurement update for a scalar measurement.

    Parameters
    ----------
    xh : np.ndarray
        State estimate, xh.shape = (n,).
    p_mat : np.ndarray
        State covariance, p_mat.shape = (n,n).
    h_vec : np.ndarray
        Measurement vector.
    r : float
        Measurement variance, scalar.
    y : float
        Measurement, scalar.

    Returns
    -------
    xh_upd : np.ndarray
        Posterior state estimate, xh_upd.shape = (n,).
    p_mat_upd : np.ndarray
        Posterior state estimate covariance, p_mat_upd.shape = (n, n).
    
    """
    m_vec = np.dot(p_mat, h_vec)
    s = np.dot(h_vec, m_vec) + r
    k_vec = m_vec / s

    yh = np.dot(h_vec, xh)

    xh_upd = xh + k_vec * (y - yh)
    p_mat_upd = p_mat - np.outer(k_vec, k_vec) * s
    p_mat_upd = 0.5 * (p_mat_upd + p_mat_upd.T)

    return xh_upd, p_mat_upd


def kf_time_linear(xh, p_mat, f_mat, g_mat, vh, q_mat):
    """
    Linear KF time update for a state estimate and covariance
    given a state transition matrix, a process noise input matrix,
    a process noise estimate and the process noise covariance.

    All inputs are numpy arrays.
    The state dimension is an integer n.
    The process noise dimension is an integer m.

    There are no dimension checks.

    Parameters
    ----------
    xh : np.ndarray
        State estimate, shape of xh = (n,).
    p_mat : np.ndarray
        State estimate covariance, p_mat, shape = (n, n).
    f_mat : np.ndarray
        State transition matrix, f_mat, shape = (n, n).
    g_mat : np.ndarray
        Process noise input matrix, g_mat, shape = (n, m).
    vh : np.ndarray
        Process noise input, shape of vh = (m, )
    q_mat : np.ndarray
        Process noise covariance matrix q_mat, shape = (m, m).

    Returns
    -------
    xh_upd : np.ndarray
        Predicted state estimate, xh_upd.shape = (n,).
    p_mat_upd : np.ndarray
        Predicted state estimate covariance, p_mat_upd.shape = (n, n).

    """
    xh_upd = np.dot(f_mat, xh) + np.dot(g_mat, vh)
    p_mat_upd = np.linalg.multi_dot((f_mat, p_mat, f_mat.T)) + np.linalg.multi_dot((g_mat, q_mat, g_mat.T))

    return xh_upd, p_mat_upd


def kf_meas_linear(xh, p_mat, h_mat, r_mat, y):
    """
    Linear KF measurement update for an a-priori state estimate and covariance
    given a measurement, the measurement matrix, and the measurement covariance.

    All inputs are numpy arrays.
    The state dimension is an integer n.
    The measurement dimension is an integer m.

    There are no dimension checks.

    Parameters
    ----------
    xh : np.ndarray
        State estimate, xh.shape = (n,).
    p_mat : np.ndarray
        State estimate covariance, p_mat.shape = (n, n).
    h_mat : np.ndarray
        Measurement matrix, h_mat.shape = (m, n).
    r_mat : np.ndarray
        Measurement noise covariance, r_mat.shape = (m, m).
    y : np.ndarray
        Measurement vector, y.shape = (m,).

    Returns
    -------
    xh_upd : np.ndarray
        Posterior state estimate, xh_upd.shape = (n,).
    p_mat_upd : np.ndarray
        Posterior state estimate covariance, p_mat_upd.shape = (n, n).
    res : np.ndarray
        Measurement residual, res.shape = (m, ).

    """
    yh = np.dot(h_mat, xh)
    res = y - yh

    m_mat = np.dot(p_mat, h_mat.T)
    s_mat = np.linalg.multi_dot((h_mat, p_mat, h_mat.T)) + r_mat
    k_mat = kf_gain(m_mat, s_mat)

    xh_upd = xh + np.dot(k_mat, res)
    p_mat_upd = p_mat - np.linalg.multi_dot((k_mat, s_mat, k_mat.T))

    return xh_upd, p_mat_upd, res


class OfflineMeasurement:
    """
    Class for handling measurements for offline Kalman filtering.

    Attributes
    ----------
    df : pd.Dataframe
        A dataframe of the data. Size m.
    time_meas : np.ndarray
        Measurement time vector. Same unit as time_reference. Size m.
    time_reference : np.ndarray
        Reference time vector for processing the measurements (e.g. in a Kalman filter). Size n.
    ind_meas : np.ndarray
        Associated index vector of size m. Each entry refers to an index of time_reference.
    ind_reference : np.ndarray
        Associated index vector of size n. Each entry refers to an index of time_meas.
    keydict : dict
        Dictionary. The keys determine the values for which measurements are generated.
        The values determine the column of df.

    Raises
    ------

    Examples
    --------
    >>> df = pd.DataFrame(data={"times": np.arange(10), "column_random": np.random.randn(10),
     "column_constant": 5*np.ones(10)}) # 1 Hz
    >>> times_reference = np.arange(0, 10, 0.1) # 10 Hz
    >>> m = OfflineMeasurement(df, df["times"], times_reference, keydict={
    "rnd": "column_random", "const": "column_constant"})
    >>> m.get_measurement(1) # None, no associated data available
    >>> m.get_measurement(0) # {'rnd': 1.1260119283350145, 'const': 5.0} (the rnd value changes)
    """

    def __init__(self, df_meas: pd.DataFrame, time_meas: np.ndarray, time_reference: np.ndarray, keydict: dict):
        """

        Parameters
        ----------
        df_meas : pd.dataframe
            A dataframe of the data. Size m.
        time_meas : np.ndarray
            Measurement time vector. Same unit as time_reference. Size m.
        time_reference : np.ndarray
            Reference time vector for processing the measurements (e.g. in a Kalman filter). Size n.
        keydict : dict
            Dictionary. The keys determine the values for which measurements are generated.
            The values determine the column of df.
        """

        self.df = df_meas

        self.time_reference = time_reference

        self.time_meas = time_meas

        self.ind_meas, _ = time_association(time_reference, time_meas)

        self.ind_reference = -np.ones(time_reference.size, dtype=int)
        self.ind_reference[self.ind_meas] = np.arange(self.ind_meas.size)

        self.keydict = keydict

    def get_measurement(self, k):
        """
        Returns a measurement dictionary for the time step k

        Parameters
        ----------
        k : int
            Integer time step of a vector with time steps.

        Returns
        -------
        meas_dict : dict
            Dictionary with measurement values for time step k.

        """
        kk = self.ind_reference[k]

        if kk == -1:
            return None

        vals = [self.df[col].iloc[kk] for col in self.keydict.values()]
        meas_dict = dict(zip(self.keydict.keys(), vals))

        return meas_dict

    def get_series(self, key):
        """
        Returns all values for a key from self.keydict

        Parameters
        ----------
        key : str
            String key for a scalar component of the measurement dictionary.

        Returns
        -------
        np.ndarray
            Numpy array with the time series of the scalar component.

        """
        return self.df[self.keydict[key]].values


class GeorefKF:

    """
    Class for offline Kalman filtering (without intermediate direction changes).

    """
    def __init__(self, params: dict):
        """


        Parameters
        ----------
        params : dict
            Dictionary containing the Kalman filter parameters. Must contain the keys 'railway_path' whose value is a
            RailwayPath object (of the selected path of the vehicle), 'forward_journey', whose value is +-1 indicating
            if the vehicle is driving forward or backward, 'tsamp', whose value is the sampling time (difference between
            consecutive timestamps) of the Kalman filter.

        Returns
        -------
        None.

        """
        self.rpath = params['railway_path']
        self.forward_journey = params['forward_journey']
        self.tsamp = params['tsamp']

        self.fmat = np.array([[1, self.tsamp], [0, 1]])
        self.gmat = np.array([self.tsamp ** 2 / 2, self.tsamp])
        self.gqgmat = np.outer(self.gmat, self.gmat)

        self.hmat_speed = np.array([0, 1])
        self.hmat_pos = np.zeros((2, 2))
        self.xest_prev = None

    def cov_symmetry(self):
        """
        Assuring that the covariance matrix is symmetric.

        Returns
        -------
        None.

        """
        self.cov = 0.5 * (self.cov + self.cov.T)

    def initialisation(self, params: dict):
        """
        Initialise the Kalman filter (setting the state vector and covariance matrix at time t=0).

        Parameters
        ----------
        params : dict
            Contains the initial state and covariance matrix (keys 'xest0', 'cov0').

        Returns
        -------
        None.

        """

        self.xest = params['xest0']
        self.cov = params['cov0']

    def time_update(self, params: dict):
        """
        Perform a time update.

        Parameters
        ----------
        params : dict
            Input for the time update. Key 'acc' contains the acceleration value for the update, 'acc_bias' the
            acceleration bias which is subtracted from the acceleration. Key 'acc_var' contains the acceleration
            variance (process noise).

        Returns
        -------
        None.

        """
        self.xest_prev = np.copy(self.xest)
        self.xest = self.fmat @ self.xest + self.gmat * (params['acc'] - params['acc_bias'])
        self.cov = self.fmat @ self.cov @ self.fmat.T + self.gqgmat * params['acc_var']

    def mu_speed(self, params: dict):
        """


        Parameters
        ----------
        params : dict
            Parameter dictionary for a speed measurement update. Must contain keys speed (measured speed value) and
            speed_var (speed variance / variance of measurement noise, e.g. provided by receiver).

        Returns
        -------
        None.

        """
        self.xest_prev = np.copy(self.xest)
        self.xest, self.cov = kf_meas_scalar(self.xest, self.cov,
                                             self.hmat_speed,
                                             params['speed_var'],
                                             params['speed'] * self.forward_journey)

    def mu_pos(self, params: dict):
        """


        Parameters
        ----------
        params : dict
            Parameter dictionary for position measurement update. Must contain keys 'pos' (input position value),
            'pos_var' (variance of measurement noise), 'dpert' (perturbation process noise).

        Returns
        -------
        None.

        """

        yest = point2array(self.rpath.path2xy(self.xest[0]))

        ypert1 = point2array(self.rpath.path2xy(self.xest[0] + 0.5 * params['dpert']))
        ypert2 = point2array(self.rpath.path2xy(self.xest[0] - 0.5 * params['dpert']))

        self.hmat_pos[:, 0] = (ypert1 - ypert2) / params['dpert']  # numerical jacobian

        m_mat = self.cov @ self.hmat_pos.T
        s_mat = self.hmat_pos @ m_mat + np.eye(2) * params['pos_var']

        k_gain = kf_gain(m_mat, s_mat)

        res = point2array(params['pos']) - yest
        self.xest_prev = np.copy(self.xest)  # only for assuring monotonicity in the distance estimates
        self.xest = self.xest + k_gain @ res

        self.cov = self.cov - np.linalg.multi_dot((k_gain, s_mat, k_gain.T))

    def fix_state(self):
        """
        This function fixes the state after updates in the sense that the speed is forced to have unique sign and the
        distance to be monotonically in(de)creasing. It is assumed that the offline KF is performed per journey,
        i.e., without intermediate direction changes.

        Parameters
        ----------


        Returns
        -------
        None.

        """
        self.xest[0] = np.maximum(self.xest[0], 0)
        # to enforce distance to be monotonically in-/decreasing
        if self.xest_prev is not None:
            distance_diff = self.xest[0] - self.xest_prev[0]
            if (self.forward_journey < 0 < distance_diff) or (self.forward_journey > 0 > distance_diff):
                self.xest[0] = self.xest_prev[0]
                self.cov[0, 0] = max(self.cov[0, 0], distance_diff**2)
        # enforce velocity to have unique sign
        if (self.forward_journey < 0 < self.xest[1]) or (self.forward_journey > 0 > self.xest[1]):
            self.cov[1, 1] = max(self.cov[1, 1], self.xest[1]**2)
            self.xest[1] = 0


def prepare_kf(params: dict, tu_params: dict, gdf_gnss: gpd.GeoDataFrame, df_imu: pd.DataFrame, acc_col: str,
               speed_col: str):
    """
    Prepare offline Kalman filter processing.

    Parameters
    ----------
    params : dict
        Parameter dictionary, must contain keys 'railway_path' (containing a RailwayPath) and 'forward_journey'.
    tu_params : dict
        Time update parameter dictionary for the Kalman filter.
    gdf_gnss : gpd.GeoDataFrame
        The GeoDataFrame containing the GNSS data.
    df_imu : pd.DataFrame
        The DataFrame containing the IMU measurement data.
    acc_col : str
        Name of the (longitudinal) acceleration column in df_imu.
    speed_col : str
        Name of the speed column in gdf_gnss.

    Returns
    -------
    kf : GeorefKF
        Initialised Kalman filter object.
    tu_params : dict
        Updated time update parameter dictionary.

    """
    acc_bias = np.mean(df_imu[acc_col])
    _, _, d0 = params['railway_path'].points2path(gdf_gnss.iloc[0].geometry)
    s0 = params['forward_journey'] * gdf_gnss[speed_col].iloc[0]
    init_params = {'xest0': np.array([d0, s0]), 'cov0': np.eye(2)}
    tu_params['acc_bias'] = acc_bias
    kf = GeorefKF(params)
    kf.initialisation(init_params)
    return kf, tu_params


def run_offline_kf(time_kf: np.ndarray, kf: GeorefKF, params: dict, tu_params: dict, mu_params: dict,
                   mm_imu: OfflineMeasurement, mm_gnss: OfflineMeasurement, standstill_kf: np.ndarray = None,
                   kf_keys: dict = {'acc': 'acc_x', 'speed': 'speed', 'pos': 'geometry', 'speed_std': 'speedacc',
                            'pos_std': 'accuracy_horizontal', 'nsat': 'nsat'}):
    """
    Runs an offline Kalman filter. The function is tailored to the available measurement quantities in the project
    OnboardEU and the default for the parameter kf_keys to the respective column naming but can easily be adapted.

    Parameters
    ----------
    time_kf : np.ndarray
        Array with timestamps of the Kalman filter. Shape (n,).
    kf : GeorefKF
        An initialized Kalman filter. Initial position and speed must already have been set.
    params : dict
        Contains parameters for running the Kalman filter. Must contain key 'acc_bias' with the acceleration bias
        (for the whole journey) as value.
    tu_params : dict
        Contains time update parameters for the Kalman filter. Must contain the key 'acc_var' containing a value for
        the variance of the acceleration values.
    mu_params : dict
        Contains measurement update parameters for the Kalman filter. Must contain the keys 'nsat_min' (minimum
        required number of visible satellites to use the GNSS data), 'accuracy_horizontal_max' (minimal required
        horizontal accuracy (as specified by the receiver) to use the GNSS data), 'dpert' (perturbation).
    mm_imu : OfflineMeasurement
        Contains the IMU measurement data of the journey.
    mm_gnss : OfflineMeasurement
        Contains the GNSS measurement data of the journey.
    standstill_kf: np.ndarray, optional
        Boolean array with flags indicating standstill. Referring to the KF timestamps (time_kf). The default is None.
    kf_keys: dict
        Specifies the keys that were used in the OfflineMeasurement objects for different quantities which will be used
        in the Kalman filter.
        Key meanings: 
            'acc': The key in the IMU OfflineMeasurement object mm_imu which contains the longitudinal acceleration
            that will be used for the time and measurement updates. Can e.g. be specified to contain the raw or filtered
            acceleration values. The default is 'acc_x'.
            'speed': Key used in the GNSS OfflineMeasurement for speed values. The default is 'speed'.
            'pos': Key used in the GNSS OfflineMeasurement for position values. The default is 'geometry'.
            'speed_std': Key used in the GNSS OfflineMeasurement for speed standard deviation values.
             The default is 'accuracy_horizontal'.
            'pos_std': Key used in the GNSS OfflineMeasurement for position standard deviation values.
             The default is 'accuracy_horizontal'.
            'nsat': Key used in the GNSS OfflineMeasurement for visible number of satellites during measurement.
             The default is 'nsat'.
         

    Returns
    -------
    est_filt : np.ndarray
        Shape (n, 2). Array containing the filtering estimates for each timestamp.
    cov_filt : np.ndarray
        Shape (n, 2, 2). Array containing the covariances for each filtering step.
    est_pred : np.ndarray
        Shape (n, 2). Array containing the prediction estimates for each timestamp.
    cov_pred : np.ndarray
        AShape (n, 2, 2). Array containing the covariances for each prediction step.

    """
    n_steps = time_kf.size
    est_filt = np.zeros((n_steps, 2))
    est_pred = 0.0 * est_filt
    cov_filt = np.zeros((n_steps, 2, 2))
    cov_pred = 0.0 * cov_filt
    if standstill_kf is None:
        standstill_kf = np.zeros(n_steps).astype(bool)
    for k, _ in enumerate(time_kf[:]):

        if k == 0:
            est_filt[k] = kf.xest
            cov_filt[k] = kf.cov

        imu_meas = mm_imu.get_measurement(k)

        if imu_meas and standstill_kf[k] == False:
            tu_params.update({'acc': imu_meas[kf_keys['acc']], 'acc_bias': params['acc_bias']})

        else:
            tu_params.update({'acc': 0, 'acc_bias': 0})
        kf.time_update(tu_params)
        kf.cov_symmetry()
        kf.fix_state()

        est_pred[k] = kf.xest
        cov_pred[k] = kf.cov

        gnss_meas = mm_gnss.get_measurement(k)

        if gnss_meas:

            if (gnss_meas[kf_keys['nsat']] < mu_params['nsat_min']
                    or gnss_meas[kf_keys['pos_std']] > mu_params['accuracy_horizontal_max']):
                pass
            else:
                mu_params.update({'speed': gnss_meas[kf_keys['speed']],
                                  'speed_var': 3 * gnss_meas[kf_keys['speed_std']] ** 2
                                  })
                kf.mu_speed(mu_params)
                kf.cov_symmetry()
                kf.fix_state()

                mu_params.update({'pos': gnss_meas[kf_keys['pos']],
                                  'pos_var': gnss_meas[kf_keys['pos_std']] ** 2
                                  })
                kf.mu_pos(mu_params)
                kf.cov_symmetry()
                kf.fix_state()
        est_filt[k] = kf.xest
        cov_filt[k] = kf.cov

    return est_filt, cov_filt, est_pred, cov_pred


def rts(xh_filt: np.ndarray, p_filt: np.ndarray, xh_pred: np.ndarray, p_pred: np.ndarray, f_mat: np.ndarray):
    """
    Performs Rauch-Tung-Striebel smooting on KF results.

    Parameters
    ----------
    xh_filt : np.ndarray
        Trajectory of state estimates.
        Filtering results that incorporate the respective measurements.
        xh_filt.shape = (nk, nx).
    p_filt : np.ndarray
        State estimate covariance matrices for xh_filt.
        p_filt.shape = (nk, nx, nx)
    xh_pred : np.ndarray
        Trajectory of state estimates.
        One-step-ahead prediction results.
        xh_pred.shape = (nk, nx)
    p_pred : np.ndarray
        State estimate covariance matrices for xh_pred.
        p_pred.shape = (nk, nx, nx).
    f_mat : np.ndarray
        State transition matrix.
        f_mat.shape = (nx, nx)

    Returns
    -------
    xh_smth : np.ndarray
        Trajectory of state estimates.
        RTS smoothing results.
        xh_pred.shape = (nk, nx)
    p_smth : np.ndarray
        State estimate covariance matrices for xh_smth.
        p_smth.shape = (nk, nx, nx).

    """

    xh_smth = xh_filt.copy()
    p_smth = p_filt.copy()

    n_steps = xh_filt.shape[0]

    for k in range(n_steps - 1)[::-1]:
        sm_gain = np.dot(np.dot(p_filt[k, :, :], f_mat.T), np.linalg.inv(p_pred[k + 1, :, :]))
        xh_smth[k] = xh_filt[k] + np.dot(sm_gain, xh_smth[k + 1] - xh_pred[k + 1])

        p_smth[k, :, :] = p_filt[k, :, :] + np.dot(
            sm_gain, np.dot(p_smth[k + 1, :, :] - p_pred[k + 1, :, :], sm_gain.T))

    return xh_smth, p_smth


def create_georef_output(time_kf: np.ndarray, est_smth: np.ndarray, cov_smth: np.ndarray, rpath: RailwayPath,
                         dec_fac: int = 1):
    """
    Create an output GeoDataFrame which contains results (geometries, speed, velocity, path distance, uncertainties,
    tracks and track distances) of offline positioning.

    Parameters
    ----------
    time_kf : np.ndarray
        Timestamps of the Kalman filter. Shape (len(time_kf), )
    est_smth : np.ndarray
        Output vector of filtering (and smoothing). Shape (len(time_kf), 2).
        Must contain distance estimates in the first and velocity estimates in the second column.
    cov_smth : np.ndarray
        Covariance estimates after filtering and (smoothing). shape=(len(time_kf), 2, 2)
    rpath : RailwayPath
        A RailwayPath object that contains the path information of the journey.
    dec_fac : int, optional
        Decimation factor. The default is 1.

    Returns
    -------
    gdf_georef : gpd.GeoDataFrame
        A GeoDataFrame containing the results of the offline positioning.

    """
    df_georef = pd.DataFrame({'time_sec': time_kf[::dec_fac]})
    df_georef['velocity'] = est_smth[::dec_fac, 1]
    df_georef['speed'] = np.absolute(est_smth[::dec_fac, 1])
    df_georef['distance_on_path'] = est_smth[::dec_fac, 0]
    df_georef['speed_std'] = np.sqrt(cov_smth[::dec_fac, 1, 1])
    df_georef['pos_std'] = np.sqrt(cov_smth[::dec_fac, 0, 0])
    geoms = []
    for k in range(0, est_smth.shape[0], dec_fac):
        geoms.append(rpath.path2xy(est_smth[k, 0]))
    gdf_georef = gpd.GeoDataFrame(df_georef, geometry=geoms)
    gdf_georef.set_crs(rpath.railway_map.geo_df.crs, inplace=True)
    gdf_georef.to_crs(epsg=4326, inplace=True)
    gdf_georef['lon'] = [pt.coords[0][0] for pt in gdf_georef.geometry]
    gdf_georef['lat'] = [pt.coords[0][1] for pt in gdf_georef.geometry]
    gdf_georef.to_crs(rpath.railway_map.geo_df.crs, inplace=True)
    tups = []
    for d in est_smth[::dec_fac, 0]:
        tups.append(rpath.path2track(d))
    gdf_georef['track_id'] = [tup[1] for tup in tups]
    gdf_georef['track_dist'] = [tup[0] for tup in tups]
    return gdf_georef


if __name__ == '__main__':
    pass
