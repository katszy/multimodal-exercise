import pandas as pd
import os
from glob import glob

#---------------------------------------------------------------------------------#
#  First step in data processing: reading the data, labelling and organizing it   #
#---------------------------------------------------------------------------------#


def rename_columns(df, suffix):
    return df.rename(columns={col: f"{col} {suffix}" for col in df.columns})

def convert_dtypes(df):
    # converts all columns in the DataFrame to float32 except 'Frame' and 'Marker'
    for col in df.columns:
        if col == "Frame":
            df[col] = df[col].astype('int64')
        elif col != "Marker":
            df[col] = df[col].astype('float32')
    return df


def load_excel_data(file, markers, joints, ergo_joints):
    print("Loading excel data from sheets")
    sheets = {
        "acc": ("Segment Acceleration", markers),
        "vel": ("Segment Velocity", markers),
        "ang_vel": ("Segment Angular Velocity", markers),
        "ang_acc": ("Segment Angular Acceleration", markers),
        "joint_angle": ("Joint Angles XZY", joints),
        "ergo_joint_angle": ("Ergonomic Joint Angles XZY", ergo_joints)
    }

    combined_data = []
    frame = pd.read_excel(file, sheet_name="Segment Acceleration", usecols=["Frame"])
    for sheet_name, (sheet, cols) in sheets.items():
        df = pd.read_excel(file, sheet_name=sheet, usecols=cols)
        renamed_df = rename_columns(df, sheet_name)
        combined_data.append(renamed_df)
        print(f"{sheet} done")

    combined_df = pd.concat([frame] + combined_data, axis=1)
    combined_df = convert_dtypes(combined_df)
    print(f"Finished reading {file} to dataframe")
    return combined_df

def label_data(markers_df, combined_df):
    print("Data labelling start")

    start_frame = None
    marker_name = None

    for _, row in markers_df.iterrows():
        if "start" in row["Marker Text"]:
            marker_name = row["Marker Text"].replace(" start", "")
            start_frame = row["Frame"]

        elif "end" in row["Marker Text"] and start_frame is not None:
            end_frame = row["Frame"]
            combined_df.loc[(combined_df['Frame'] >= start_frame) & (combined_df['Frame'] <= end_frame), "Marker"] = marker_name
            start_frame = None
            marker_name = None

    print("Data labelling done")
    return combined_df


def load_emg_data(file):
    print(f"Loading EMG data from {file}")
    try:
        emg = pd.read_excel(file, sheet_name="External Data")
        markers_df = pd.read_excel(file, sheet_name="Markers", usecols=["Frame", "Marker Text"])
    except ValueError as e:
        print(f"Error: 'External Data' sheet is missing from {file}")
        return None
    print("Labelling EMG data")
    labelled_emg = label_data(markers_df, emg)
    return labelled_emg

def process_session(participant, session_index, markers, joints, ergo_joints):
    """
        Reads and labels session data (all weeks) for one person.
        Markers, joints, and egro joints define which sensors and joint angles should be included.
        Output: one imu pickle file (one person all session) and one emg pickle file (one person all session)
    """
    session_files = glob(f"../data/input/{participant}/session{session_index}/*.xlsx")
    combined_data = []
    emg_data_list = []
    print(session_files)

    for file in session_files:
        print(f"Processing {file}")
        # Read IMU data
        combined_df = load_excel_data(file, markers, joints, ergo_joints)
        # Label IMU data
        labels_df = pd.read_excel(file, sheet_name="Markers", usecols=["Frame", "Marker Text"])
        labelled_df = label_data(labels_df, combined_df)
        combined_data.append(labelled_df)

        # Read and label EMG data
        emg_df = load_emg_data(file)
        emg_data_list.append(emg_df)

    # Combine session data
    session_data = pd.concat(combined_data, ignore_index=True)
    emg_data = pd.concat(emg_data_list, ignore_index=True)

    # Save IMU data as pickle
    session_folder = f"../data/raw/{participant}/session{session_index}"
    os.makedirs(session_folder, exist_ok=True)
    pickle_path = os.path.join(session_folder, f"session{session_index}.pkl")
    session_data.to_pickle(pickle_path)
    print(f"Saved session data to {pickle_path}")

    # Save EMG data as separate pickle
    pickle_path_emg = os.path.join(session_folder, f"session{session_index}_emg.pkl")
    emg_data.to_pickle(pickle_path_emg)
    print(f"Saved emg data to {pickle_path_emg}")


def get_available_sessions(participant_path):
    session_folders = [f for f in os.listdir(participant_path) if os.path.isdir(os.path.join(participant_path, f))]
    return len(session_folders)


# Main script
markers = ["Pelvis", "L5", "L3", "T12", "T8", "Right Shoulder", "Left Shoulder"]
markers = [f"{m} {axis}" for m in markers for axis in ['x', 'y', 'z']]
joints = ["L5S1 Lateral Bending", "L5S1 Axial Bending", "L5S1 Flexion/Extension",
            "L4L3 Lateral Bending", "L4L3 Axial Rotation", "L4L3 Flexion/Extension",
            "L1T12 Lateral Bending", "L1T12 Axial Rotation", "L1T12 Flexion/Extension",
            "Right Hip Abduction/Adduction", "Left Hip Abduction/Adduction",
            "Right Hip Internal/External Rotation", "Left Hip Internal/External Rotation",
            "Right Hip Flexion/Extension", "Left Hip Flexion/Extension"]
ergo_joints = ["Pelvis_T8 Lateral Bending", "Pelvis_T8 Axial Bending",
                "Pelvis_T8 Flexion/Extension","Vertical_T8 Lateral Bending",
                "Vertical_T8 Axial Bending", "Vertical_T8 Flexion/Extension"]

participants = [folder for folder in os.listdir('../data/input') if os.path.isdir(os.path.join('../../data/input', folder))]
for participant in participants:
    participant_path = f"../data/input/{participant}"
    session_numbers = get_available_sessions(participant_path)
    for session_index in range(1, session_numbers+1):
        print(f"Processing session {session_index}")
        process_session(participant, session_index, markers, joints, ergo_joints)
