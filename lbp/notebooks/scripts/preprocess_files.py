from scipy import signal
import pyemgpipeline.processors as processors
import pyemgpipeline as pep
import pandas as pd
from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt
import numpy as np

# Creating a dictionary with MVC values for each person
mvc_dict = {
    "Anita": {"mvc_left": 332.7315, "mvc_right": 255.2501},
    "Dávid": {"mvc_left": 316.8271, "mvc_right": 316.7368},
    "Flóra": {"mvc_left": 224.0826, "mvc_right": 175.1042},
    "István": {"mvc_left": 108.5914, "mvc_right": 158.3160},
    "Kitti": {"mvc_left": 278.4766, "mvc_right": 287.3040},
    "Kristóf": {"mvc_left": 294.5414, "mvc_right": 214.8884},
    "Margita": {"mvc_left": 222.7770, "mvc_right": 247.6796},
    "Patrik": {"mvc_left": 488.4035, "mvc_right": 311.5888},
    "Petra": {"mvc_left": 411.9400, "mvc_right": 344.3418},
    "Zoltán": {"mvc_left": 185.2618, "mvc_right": 181.3279}
}


def proc_emg(df, name):
    # Constants
    hz = 2000  # Sampling rate
    w0 = 50 / (hz / 2)  # Notch filter at 50 Hz
    Q = 30.0  # Quality factor for notch filter

    for col in ['left', 'right']:
        print(f"Processing column: {col}")
        data = df[col].to_numpy()

        # Notch filter
        b, a = signal.iirnotch(w0, Q)
        notch_filtered = signal.lfilter(b, a, data)

        m = pep.wrappers.EMGMeasurement(notch_filtered, hz=hz, trial_name=col)
        m.apply_dc_offset_remover()
        m.apply_bandpass_filter(bf_order=4, bf_cutoff_fq_lo=25, bf_cutoff_fq_hi=350)
        df[f'{col} bandpassed'] = m.data
        m.apply_full_wave_rectifier()
        m.apply_linear_envelope(le_order=6, le_cutoff_fq=2)

        # Apply amplitude normalization based on MVC (Maximum Voluntary Contraction)
        if col == 'left':
            mvc_left = mvc_dict[name]["mvc_left"]
            m.apply_amplitude_normalizer(mvc_left)
        else:
            mvc_right = mvc_dict[name]["mvc_right"]
            m.apply_amplitude_normalizer(mvc_right)

        df[f'{col} envelope'] = m.data

    return df


def butter_lowpass_filter(data, cutoff=3, fs=60, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def proc_imu(df):
    # Filter with low pass filter
    for column in df.columns:
        if "acc" in column:
            df[column] = butter_lowpass_filter(df[column], cutoff=5, fs=60, order=4)
        elif "vel" in column:
            df[column] = butter_lowpass_filter(df[column], cutoff=5, fs=60, order=4)

    # Calculate magnitudes
    types = ["acc", "ang_acc", "vel", "ang_vel"]
    magnitude_data = {}
    for t in types:
        axis_sets = set([col.rsplit(' ', 2)[0] for col in df.columns if f"x {t}" in col])
        for axis in axis_sets:
            x_col = f"{axis} x {t}"
            y_col = f"{axis} y {t}"
            z_col = f"{axis} z {t}"
            if x_col in df.columns and y_col in df.columns and z_col in df.columns:
                magnitude_data[f"{axis} {t} magnitude"] = np.sqrt(df[x_col] ** 2 + df[y_col] ** 2 + df[z_col] ** 2)
    magnitude_df = pd.DataFrame(magnitude_data)
    df = pd.concat([df, magnitude_df], axis=1)

    return df


