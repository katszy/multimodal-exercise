from glob import glob
import pandas as pd
import os

# --------------------------------------------------------------
# Read sheet names
# --------------------------------------------------------------

xls = pd.ExcelFile("data/input/Participant_A.xlsx")
sheet_names = xls.sheet_names
print(sheet_names)

# --------------------------------------------------------------
# List all data in data/in
# --------------------------------------------------------------

files_correct = glob("data/in/correct/*.xlsx")
print(len(files_correct))
files_incorrect = glob("data/in/incorrect/*.xlsx")
print(len(files_incorrect))

# --------------------------------------------------------------
# Define columns we need
# --------------------------------------------------------------

relevant_markers = ["Frame", "Left Shoulder x", "Left Shoulder y", "Left Shoulder z",
                    "Left Upper Arm x", "Left Upper Arm y", "Left Upper Arm z",
                    "Left Forearm x", "Left Forearm y", "Left Forearm z",
                    "Left Hand x", "Left Hand y", "Left Hand z"]

relevant_joints = ['Left Shoulder Abduction/Adduction',
                   'Left Shoulder Flexion/Extension',
                   'Left Elbow Flexion/Extension']

relevant_ergo_joints = ['T8_LeftUpperArm Lateral Bending', 'T8_LeftUpperArm Flexion/Extension']


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

def load_data(excel_path, sheet_name, columns=None):
    print(f"Loading data from Excel file.{excel_path}")
    data = pd.read_excel(excel_path, sheet_name=sheet_name, usecols=columns)
    return data


def get_dataframe(files, folder):
    for f in files:
        file_name = os.path.basename(f)
        participant = file_name.split('.')[0]

        acc = load_data(f, "Segment Acceleration", relevant_markers)
        def drop_frame_column(df):
            # This function tries to drop the 'Frame' column if it exists.
            if 'Frame' in df.columns:
                return df.drop(columns='Frame')
            else:
                return df

        vel = drop_frame_column(load_data(f, "Segment Velocity", relevant_markers))
        ang_vel = drop_frame_column(load_data(f, "Segment Angular Velocity", relevant_markers))
        ang_acc = drop_frame_column(load_data(f, "Segment Angular Acceleration", relevant_markers))
        position = drop_frame_column(load_data(f, "Segment Position", relevant_markers))
        joint_angle_xzy = drop_frame_column(load_data(f, "Joint Angles XZY", relevant_joints))
        ergonomic_joint_angle_xzy = drop_frame_column(load_data(f, "Ergonomic Joint Angles XZY", relevant_ergo_joints))

        # Add suffix to DataFrame columns to reflect their origin
        acc = acc.rename(columns={col: f"{col} acc" for col in acc.columns})
        acc = acc.rename(columns={'Frame acc': 'Frame'})
        vel = vel.rename(columns={col: f"{col} vel" for col in vel.columns})
        ang_vel = ang_vel.rename(columns={col: f"{col} ang_vel" for col in ang_vel.columns})
        ang_acc = ang_acc.rename(columns={col: f"{col} ang_acc" for col in ang_acc.columns})
        position = position.rename(columns={col: f"{col} position" for col in position.columns})
        joint_angle_xzy = joint_angle_xzy.rename(
            columns={col: f"{col} joint_angle_xzy" for col in joint_angle_xzy.columns})
        ergonomic_joint_angle_xzy = ergonomic_joint_angle_xzy.rename(
            columns={col: f"{col} joint_angle_zxy" for col in ergonomic_joint_angle_xzy.columns})

        # Combine them
        combined_df = pd.concat([acc, vel, ang_vel, ang_acc, position, joint_angle_xzy, ergonomic_joint_angle_xzy], axis=1)

        # Label them
        def label_sets(df, start_end, start_end_2, participant):
            df = df.copy()
            df['Set'] = 0
            df["Participant"] = participant
            df["Correct"] = 0

            for idx, (start, end) in enumerate(start_end, start=1):
                df.loc[start:end, "Set"] = idx
                df.loc[start:end, "Correct"] = 1

            for idx, (start, end) in enumerate(start_end_2, start=1):
                df.loc[start:end, "Set"] = idx
                df.loc[start:end, "Correct"] = 0
            df = df[df['Set'] != 0]
            return df

        combined_df["Participant"] = participant
        if folder == "correct":
            combined_df["Correct"] = 1
        elif folder == "incorrect":
            combined_df["Correct"] = 0

        pickle_file = f"data/out/{folder}/" + participant  # folder: correct or incorrect
        combined_df.to_pickle(pickle_file + ".pkl")


get_dataframe(files_correct, "correct")
get_dataframe(files_incorrect, "incorrect")

# --------------------------------------------------------------
# Read pickle files of individual reps and combine them into 2 large dataframes (correct and incorrect)
# --------------------------------------------------------------

print("Creating dataset")

correct_pickle = glob("data/out/correct/*.pkl")
correct_dataframes = [pd.read_pickle(file) for file in correct_pickle]
df_combined_correct = pd.concat(correct_dataframes, ignore_index=True)
df_combined_correct.to_pickle("data/out/imu_correct.pkl")

incorrect_pickle = glob("data/out/incorrect/*.pkl")
incorrect_dataframes = [pd.read_pickle(file) for file in incorrect_pickle]
df_combined_incorrect = pd.concat(incorrect_dataframes, ignore_index=True)
df_combined_incorrect.to_pickle("data/out/imu_incorrect.pkl")
