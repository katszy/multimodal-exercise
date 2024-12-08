import numpy as np
from dtw import *
import pandas as pd
import statsmodels.api as sm
from scipy.signal import  find_peaks

def dynamic_time_warping(x, y):
    alignment = dtw(x, y, keep_internals=True)
    distance = alignment.distance
    return alignment, distance

def dtw_reps_position(df, t=None):
    participants = df['Participant'].unique()
    s = 0
    for participant in participants:

        if t is not None:
            x = t[(t["Participant"] == participant) & (t['Set'] == 1) & (t['Repetition'] == 1)][
                'Left Hand x position'].to_numpy()
            y = t[(t["Participant"] == participant) & (t['Set'] == 1) & (t['Repetition'] == 1)][
                'Left Hand y position'].to_numpy()
            z = t[(t["Participant"] == participant) & (t['Set'] == 1) & (t['Repetition'] == 1)][
                'Left Hand z position'].to_numpy()
            template = np.column_stack((x, y, z))

        else:
            x = df[(df["Participant"] == participant) & (df['Set'] == 1) & (df['Repetition'] == 1)][
                'Left Hand x position'].to_numpy()
            y = df[(df["Participant"] == participant) & (df['Set'] == 1) & (df['Repetition'] == 1)][
                'Left Hand y position'].to_numpy()
            z = df[(df["Participant"] == participant) & (df['Set'] == 1) & (df['Repetition'] == 1)][
                'Left Hand z position'].to_numpy()
            template = np.column_stack((x, y, z))

        sets = df[(df["Participant"] == participant)]['Set'].unique()
        sum_distance = 0
        counter = 0

        for set_number in sets:
            repetitions = df[(df["Participant"] == participant) & (df['Set'] == set_number)]['Repetition'].unique()
            for rep_number in repetitions:
                data = df[
                    (df["Participant"] == participant) & (df['Set'] == set_number) & (df['Repetition'] == rep_number)]
                sample = np.column_stack(
                    (data['Left Hand x position'], data['Left Hand y position'], data['Left Hand z position']))

                # Calculate the DTW distance
                alignment = dtw(template, sample, keep_internals=True)
                distance = alignment.distance
                distance = distance / len(sample)
                sum_distance += distance
                counter += 1

        # Calculate average DTW distance
        if counter > 0:
            average = sum_distance / counter
            print(f"Average DTW distance for {participant}: {average}")
            s += average
        else:
            print(f"No data to process for {participant}")
    overall_average = s / len(participants)
    print(f"Overall average DTW distance (reps): {overall_average}")
    return overall_average

def dtw_reps_joint(df, label, t=None):
    participants = df['Participant'].unique()
    s = 0
    for participant in participants:

        if t is not None:
            template = t[(t["Participant"] == participant) & (t['Set'] == 1) & (t['Repetition'] == 1)][label].to_numpy()

        else:
            template = df[(df["Participant"] == participant) & (df['Set'] == 1) & (df['Repetition'] == 1)][
                label].to_numpy()

        sets = df[(df["Participant"] == participant)]['Set'].unique()
        sum_distance = 0
        counter = 0

        for set_number in sets:
            repetitions = df[(df["Participant"] == participant) & (df['Set'] == set_number)]['Repetition'].unique()
            for rep_number in repetitions:
                data = df[
                    (df["Participant"] == participant) & (df['Set'] == set_number) & (df['Repetition'] == rep_number)]
                sample = data[label]

                # Calculate the DTW distance
                alignment = dtw(template, sample, keep_internals=True)
                distance = alignment.distance
                distance = distance / len(sample)

                sum_distance += distance
                counter += 1

        # Calculate average DTW distance
        if counter > 0:
            average = sum_distance / counter
            print(f"Average DTW distance for {participant}: {average}")
            s += average
        else:
            print(f"No data to process for {participant}")
    overall_average = s / len(participants)
    print(f"Overall average DTW distance (reps): {overall_average}")
    return overall_average


def cosine_similarity(x, y):
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        # Return 0 similarity if either vector is zero
        return 0
    dot_product = np.dot(x, y)
    return dot_product / (norm_x * norm_y)

def cosine_similarity_time_series(ts1, ts2):
    # Pad the time series to have the same length
    if ts1.shape != ts2.shape:
        ts1, ts2 = pad_time_series(ts1, ts2)

    num_points = ts1.shape[0]
    similarities = np.zeros(num_points)

    for i in range(num_points):
        # Compute cosine similarity between corresponding vectors at each time point
        similarities[i] = cosine_similarity(ts1[i], ts2[i])

    return np.mean(similarities)

def pad_time_series(ts1, ts2):
    """
    Pads the shorter time series with zeros to match the length of the longer time series.
    Handles both 1D and 2D time series.
    """
    len1, len2 = ts1.shape[0], ts2.shape[0]

    if len1 == len2:
        return ts1, ts2

    # Determine the padding dimensions
    padding_shape = (abs(len1 - len2),) if len(ts1.shape) == 1 else (abs(len1 - len2), ts1.shape[1])

    # Add zero padding to the shorter time series
    if len1 > len2:
        ts2 = np.pad(ts2, ((0, padding_shape[0]),) + ((0, 0),) * (len(ts1.shape) - 1), mode='constant')
    else:
        ts1 = np.pad(ts1, ((0, padding_shape[0]),) + ((0, 0),) * (len(ts2.shape) - 1), mode='constant')

    return ts1, ts2


def euclidean_3d(ts1, ts2):
    """
    Computes the Euclidean distance between two 3D time series after padding the shorter one with zeros.
    """
    # Pad the time series to have the same length
    if ts1.shape != ts2.shape:
        ts1, ts2 = pad_time_series(ts1, ts2)

    # Extract individual components
    x1, y1, z1 = ts1[:, 0], ts1[:, 1], ts1[:, 2]
    x2, y2, z2 = ts2[:, 0], ts2[:, 1], ts2[:, 2]

    dist = 0
    for i in range(len(x1)):
        dx = x2[i] - x1[i]
        dy = y2[i] - y1[i]
        dz = z2[i] - z1[i]
        squared_distances = dx ** 2 + dy ** 2 + dz ** 2
        distance = np.sqrt(squared_distances)
        dist += distance
    dist = dist/len(ts1)
    return dist

def eucledian(x, y):
    if x.shape != y.shape:
        x, y = pad_time_series(x, y)

    dist = np.sqrt(np.sum([(a - b) * (a - b) for a, b in zip(x, y)]))
    dist = dist / len(x)
    return dist

def rmse(a, b):
    if a.shape != b.shape:
        a, b = pad_time_series(a, b)
    differences = a - b
    squared_differences = differences ** 2
    mean_squared_error = np.mean(squared_differences)
    root_mean_squared_error = np.sqrt(mean_squared_error)
    return root_mean_squared_error

def rmse_3d(ts1, ts2):
    if ts1.shape != ts2.shape:
        ts1, ts2 = pad_time_series(ts1, ts2)
    squared_differences = (ts1 - ts2) ** 2
    mean_squared_error = np.mean(np.sum(squared_differences, axis=1))
    root_mean_squared_error = np.sqrt(mean_squared_error)
    return root_mean_squared_error

def bring_to_same_length(data, indices):
    return data[indices]

def process(ts1, ts2):
    # Perform dynamic time warping
    alignment, distance = dynamic_time_warping(ts1, ts2)
    aligned_ts1_indices = alignment.index1
    aligned_ts2_indices = alignment.index2

    # Check if the data is 1D or 3D
    if ts1.ndim == 1:
        # 1D data processing
        aligned_1 = bring_to_same_length(ts1, aligned_ts1_indices)
        aligned_2 = bring_to_same_length(ts2, aligned_ts2_indices)
    else:
        # 3D data processing
        aligned_x1 = bring_to_same_length(ts1[:, 0], aligned_ts1_indices)
        aligned_x2 = bring_to_same_length(ts2[:, 0], aligned_ts2_indices)
        aligned_y1 = bring_to_same_length(ts1[:, 1], aligned_ts1_indices)
        aligned_y2 = bring_to_same_length(ts2[:, 1], aligned_ts2_indices)
        aligned_z1 = bring_to_same_length(ts1[:, 2], aligned_ts1_indices)
        aligned_z2 = bring_to_same_length(ts2[:, 2], aligned_ts2_indices)

        aligned_1 = np.column_stack((aligned_x1, aligned_y1, aligned_z1))
        aligned_2 = np.column_stack((aligned_x2, aligned_y2, aligned_z2))

    return aligned_1, aligned_2, distance

def calc_metrics_position(df, t=None):
    participants = df['Participant'].unique()
    s = 0
    results =[]
    for participant in participants:

        if t is not None:
            x = t[(t["Participant"] == participant) & (t['Set'] == 1) & (t['Repetition'] == 1)][
                'Left Hand x position'].to_numpy()
            y = t[(t["Participant"] == participant) & (t['Set'] == 1) & (t['Repetition'] == 1)][
                'Left Hand y position'].to_numpy()
            z = t[(t["Participant"] == participant) & (t['Set'] == 1) & (t['Repetition'] == 1)][
                'Left Hand z position'].to_numpy()
            template = np.column_stack((x, y, z))

        else:
            x = df[(df["Participant"] == participant) & (df['Set'] == 1) & (df['Repetition'] == 1)][
                'Left Hand x position'].to_numpy()
            y = df[(df["Participant"] == participant) & (df['Set'] == 1) & (df['Repetition'] == 1)][
                'Left Hand y position'].to_numpy()
            z = df[(df["Participant"] == participant) & (df['Set'] == 1) & (df['Repetition'] == 1)][
                'Left Hand z position'].to_numpy()
            template = np.column_stack((x, y, z))

        sets = df[(df["Participant"] == participant)]['Set'].unique()
        sum_distance = 0
        counter = 0

        euc_dist_dtw_sum = 0
        euc_dist_0pad_sum = 0
        cosine_dist_dtw_sum = 0
        cosine_dist_0pad_sum = 0
        rmse_dtw_sum = 0
        rmse_0pad_sum = 0

        for set_number in sets:
            repetitions = df[(df["Participant"] == participant) & (df['Set'] == set_number)]['Repetition'].unique()
            for rep_number in repetitions:
                data = df[
                    (df["Participant"] == participant) & (df['Set'] == set_number) & (df['Repetition'] == rep_number)]
                sample = np.column_stack(
                    (data['Left Hand x position'], data['Left Hand y position'], data['Left Hand z position']))

                # Calculate the DTW distance
                alignment = dtw(template, sample, keep_internals=True)
                distance = alignment.distance
                distance = distance / len(sample)

                # Bring data to same length, using dtw
                aligned1, aligned2, _ = process(template, sample)

                # Calculate Eucledian distance
                euc_dist_dtw = euclidean_3d(aligned1, aligned2)
                euc_dist_0pad = euclidean_3d(template, sample)

                # Calculate cosine similarity
                cosine_dist_dtw = cosine_similarity_time_series(aligned1, aligned2)
                cosine_dist_0pad = cosine_similarity_time_series(template, sample)

                rmse_dtw = rmse_3d(aligned1, aligned2)
                rmse_0pad = rmse_3d(template, sample)

                sum_distance += distance
                counter += 1
                euc_dist_dtw_sum += euc_dist_dtw
                euc_dist_0pad_sum += euc_dist_0pad
                cosine_dist_dtw_sum += cosine_dist_dtw
                cosine_dist_0pad_sum += cosine_dist_0pad
                rmse_dtw_sum += rmse_dtw
                rmse_0pad_sum = rmse_0pad

        # Calculate average DTW distance
        if counter > 0:
            average = sum_distance / counter
            print(f"Average EUC+DTW distance for {participant}: {euc_dist_dtw_sum/counter}")
            print(f"Average EUC+0pad distance for {participant}: {euc_dist_0pad_sum/counter}")
            print(f"Average COS+DTW distance for {participant}: {cosine_dist_dtw_sum/counter}")
            print(f"Average COS+0pad distance for {participant}: {cosine_dist_0pad_sum/counter}")
            print(f"Average RMSE+DTW distance for {participant}: {rmse_dtw_sum / counter}")
            print(f"Average RMSE+0pad distance for {participant}: {rmse_0pad_sum / counter}")
            s += average
            results.append({
                'Participant': participant,
                'Average EUC+DTW distance': euc_dist_dtw_sum / counter,
                'Average EUC+0pad distance': euc_dist_0pad_sum / counter,
                'Average COS+DTW distance': cosine_dist_dtw_sum / counter,
                'Average COS+0pad distance': cosine_dist_0pad_sum / counter,
                'Average RMSE+DTW distance': rmse_dtw_sum / counter,
                'Average RMSE+0pad distance': rmse_0pad_sum / counter
            })
        else:
            print(f"No data to process for {participant}")
    overall_average = s / len(participants)
    results_df = pd.DataFrame(results)
    return results_df

def calc_metrics_joint(df, label, t=None):
    participants = df['Participant'].unique()
    s = 0
    results = []
    for participant in participants:

        if t is not None:
            template = t[(t["Participant"] == participant) & (t['Set'] == 1) & (t['Repetition'] == 1)][label].to_numpy()

        else:
            template = df[(df["Participant"] == participant) & (df['Set'] == 1) & (df['Repetition'] == 1)][label].to_numpy()

        sets = df[(df["Participant"] == participant)]['Set'].unique()
        sum_distance = 0
        counter = 0
        euc_dist_dtw_sum = 0
        euc_dist_0pad_sum = 0
        cosine_dist_dtw_sum = 0
        cosine_dist_0pad_sum = 0
        rmse_dtw_sum = 0
        rmse_0pad_sum = 0

        for set_number in sets:
            repetitions = df[(df["Participant"] == participant) & (df['Set'] == set_number)]['Repetition'].unique()
            for rep_number in repetitions:
                data = df[(df["Participant"] == participant) & (df['Set'] == set_number) & (df['Repetition'] == rep_number)]
                sample = data[label].values

                # Calculate the DTW distance
                alignment = dtw(template, sample, keep_internals=True)
                distance = alignment.distance
                distance = distance / len(sample)

                # Bring data to same length, using dtw
                aligned1, aligned2, _ = process(template, sample)

                # Calculate Eucledian distance
                euc_dist_dtw = eucledian(aligned1, aligned2)
                euc_dist_0pad = eucledian(template, sample)

                # Calculate cosine similarity
                cosine_dist_dtw = cosine_similarity_time_series(aligned1, aligned2,)
                cosine_dist_0pad = cosine_similarity_time_series(template, sample)

                rmse_dtw = rmse(aligned1, aligned2)
                rmse_0pad = rmse(template, sample)

                counter += 1
                euc_dist_dtw_sum += euc_dist_dtw
                euc_dist_0pad_sum += euc_dist_0pad
                cosine_dist_dtw_sum += cosine_dist_dtw
                cosine_dist_0pad_sum += cosine_dist_0pad
                rmse_dtw_sum += rmse_dtw
                rmse_0pad_sum = rmse_0pad
                sum_distance += distance
                counter += 1

        # Calculate average DTW distance
        if counter > 0:
            average = sum_distance / counter
            print(f"Average EUC+DTW distance for {participant}: {euc_dist_dtw_sum / counter}")
            print(f"Average EUC+0pad distance for {participant}: {euc_dist_0pad_sum / counter}")
            print(f"Average COS+DTW distance for {participant}: {cosine_dist_dtw_sum / counter}")
            print(f"Average COS+0pad distance for {participant}: {cosine_dist_0pad_sum / counter}")
            print(f"Average RMSE+DTW distance for {participant}: {rmse_dtw_sum / counter}")
            print(f"Average RMSE+0pad distance for {participant}: {rmse_0pad_sum / counter}")
            s += average
            results.append({
                'Participant': participant,
                'Average EUC+DTW distance': euc_dist_dtw_sum / counter,
                'Average EUC+0pad distance': euc_dist_0pad_sum / counter,
                'Average COS+DTW distance': cosine_dist_dtw_sum / counter,
                'Average COS+0pad distance': cosine_dist_0pad_sum / counter,
                'Average RMSE+DTW distance': rmse_dtw_sum / counter,
                'Average RMSE+0pad distance': rmse_0pad_sum / counter
            })
        else:
            print(f"No data to process for {participant}")
    overall_average = s / len(participants)
    results_df = pd.DataFrame(results)
    return results_df



def estimated_autocorrelation(x):
    result = sm.tsa.acf(x, nlags=500)

    peaks, _ = find_peaks(result)
    peak_values = result[peaks]
    valid_peaks = peaks[peaks > 50]
    valid_peak_values = result[valid_peaks]

    # Check if there are valid peaks and find the largest or second largest as needed
    if len(valid_peak_values) == 0:
        return result, None, None

    if 1 in valid_peak_values:
        second_largest_value = np.partition(valid_peak_values, -2)[-2]
        second_largest_index = valid_peaks[np.argsort(valid_peak_values)[-2]]
    else:
        second_largest_value = valid_peak_values.max()
        second_largest_index = valid_peaks[valid_peak_values.argmax()]

    return result, second_largest_value, second_largest_index


def max_peak_autocorr(df, label):
    participants = df["Participant"].unique()
    sets = df["Set"].unique()

    sum = 0
    counter = 0
    results=[]
    for participant in participants:
        su = 0
        c = 0
        for s in sets:
            data = df[(df["Participant"] == participant) & (df['Set'] == s)][label]
            if not data.empty:
                autocorr, max_peak_value, max_peak_index = estimated_autocorrelation(data)
                counter += 1
                c += 1
                su += max_peak_value
                sum += max_peak_value
                print(f"Participant: {participant}, Max Peak Value: {max_peak_value}, index: {max_peak_index}")
        print(f"Participant average: {su/c}")
        results.append({
            'Participant': participant,
            'Max Peak Value Average': su/c
        })
    results_df = pd.DataFrame(results)
    return results_df

def movement_intensity(x, y, z, g=9.81):
    acceleration_norms = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    average_mi = np.mean(acceleration_norms / g)
    return average_mi


def log_dimensionless_jerk(movement, fs=2000):
    """
     Calculates the smoothness metric for the given speed profile using the
    log dimensionless jerk metric.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.

    Returns
    -------
    dl       : float
               The dimensionless jerk estimate of the given movement's
               smoothness.

    """

    movement = np.array(movement)
    movement_peak = max(abs(movement))
    dt = 1. / fs
    movement_dur = len(movement) * dt
    jerk = np.diff(movement, 2) / pow(dt, 2)
    scale = pow(movement_dur, 3) / pow(movement_peak, 2)
    dlj = - scale * sum(pow(jerk, 2)) * dt
    return -np.log(abs(dlj))


def sparc(movement, fs=60, padlevel=4, fc=3.0, amp_th=0.05):
    """
        Calcualtes the smoothness of the given speed profile using the modified
        spectral arc length metric.

        Parameters
        ----------
        movement : np.array
                   The array containing the movement speed profile.
        fs       : float
                   The sampling frequency of the data.
        padlevel : integer, optional
                   Indicates the amount of zero padding to be done to the movement
                   data for estimating the spectral arc length. [default = 4]
        fc       : float, optional
                   The max. cut off frequency for calculating the spectral arc
                   length metric. [default = 10.]
        amp_th   : float, optional
                   The amplitude threshold to used for determing the cut off
                   frequency upto which the spectral arc length is to be estimated.
                   [default = 0.05]

        Returns
        -------
        sal      : float
                   The spectral arc length estimate of the given movement's
                   smoothness.
        (f, Mf)  : tuple of two np.arrays
                   This is the frequency(f) and the magntiude spectrum(Mf) of the
                   given movement data. This spectral is from 0. to fs/2.
        (f_sel, Mf_sel) : tuple of two np.arrays
                          This is the portion of the spectrum that is selected for
                          calculating the spectral arc length.

        Notes
        -----
        This is the modfieid spectral arc length metric, which has been tested only
        for discrete movements.

        Examples
        --------
        >>> t = np.arange(-1, 1, 0.01)
        >>> move = np.exp(-5*pow(t, 2))
        >>> sal, _, _ = sparc(move, fs=100.)
        >>> '%.5f' % sal
        '-1.41403'

        """
    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    # Frequency
    f = np.arange(0, fs, fs / nfft)
    # Normalized magnitude spectrum
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf / max(Mf)

    fc_inx = ((f <= fc) * 1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate arc length
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) + pow(np.diff(Mf_sel), 2)))
    return new_sal, (f, Mf), (f_sel, Mf_sel)