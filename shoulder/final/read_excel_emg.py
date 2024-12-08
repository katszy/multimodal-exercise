import os
import pandas as pd
from scipy import signal
import pyemgpipeline.processors as processors


def load_data(excel_path, sheet_name, pickle_path, columns=None):
    if os.path.exists(pickle_path + "_" + sheet_name + ".pkl"):
        print(f"Loading data from pickle file {pickle_path + '_' + sheet_name + '.pkl'}.")
        return pd.read_pickle(pickle_path + "_" + sheet_name + ".pkl")
    else:
        print(f"Loading data from Excel file {excel_path}, sheet: {sheet_name}.")
        if columns:
            data = pd.read_excel(excel_path, sheet_name=sheet_name, usecols=columns)
        else:
            data = pd.read_excel(excel_path, sheet_name=sheet_name)
        print(f"Writing data to pickle file {pickle_path + '_' + sheet_name + '.pkl'}.")
        data.to_pickle(pickle_path + "_" + sheet_name + ".pkl")
        return data


def preprocess_signals(df_raw):
    # Constants
    hz = 2000
    fs = 2000  # Sampling rate
    w0 = 50 / (fs / 2)
    Q = 30.0

    for col in df_raw.columns:
        if col in ['Sample', 'Frame']:
            continue

        print(f"Processing column: {col}")
        unfiltered = df_raw[col].to_numpy()
        dc_remover = processors.DCOffsetRemover()
        signal_without_dc = dc_remover.apply(unfiltered)

        bandpass_filter = processors.BandpassFilter(hz=hz, bf_order=4, bf_cutoff_fq_lo=25, bf_cutoff_fq_hi=350)
        bandpassed_filtered = bandpass_filter.apply(signal_without_dc)

        b, a = signal.iirnotch(w0, Q)
        notch_filtered = signal.lfilter(b, a, bandpassed_filtered)

        rectifier = processors.FullWaveRectifier()
        rectified_signal = rectifier.apply(notch_filtered)

        linear_envelope_processor = processors.LinearEnvelope(hz=hz, le_order=4, le_cutoff_fq=6)
        envelope_signal = linear_envelope_processor.apply(rectified_signal)

        amplitude_normalizer = processors.AmplitudeNormalizer()
        normalized_signal = amplitude_normalizer.apply(rectified_signal, divisor=rectified_signal.max())

        # Add the new columns directly to the DataFrames
        df_raw[f'{col} filtered'] = normalized_signal
        df_raw[f'{col} envelope'] = envelope_signal

    return df_raw


def process_participant_files(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    combined_df = pd.DataFrame()

    for file_path in files:
        file_name = os.path.basename(file_path)
        participant = file_name.split('.')[0]
        pickle_path = os.path.join(folder_path, participant)

        df_raw = load_data(file_path, "External Data", pickle_path)
        df_processed = preprocess_signals(df_raw)
        df_processed['Participant'] = participant

        combined_df = pd.concat([combined_df, df_processed], ignore_index=True)

    return combined_df


data_folder = "data/in/correct"
combined_df = process_participant_files(data_folder)
print(combined_df.head())
combined_df.to_pickle("data/out/emg_correct.pkl")

data_folder = "data/in/incorrect"
combined_df_i = process_participant_files(data_folder)
print(combined_df_i.head())
combined_df_i.to_pickle("data/out/emg_correct.pkl")
