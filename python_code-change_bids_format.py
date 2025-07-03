# @title change the format to BIDS format
import os
import mne
import shutil
import pandas as pd
from mne_bids import write_raw_bids, BIDSPath
from pathlib import Path

# Set your input and output directories
input_dir = Path("/home/hengjie/Downloads/alzheimer_data-nature/EEG_2") # directory of the raw dataset
bids_root = Path("/home/hengjie/Downloads/alzheimer_data-nature/bids_dataset") # directory of where the data is saved
temp_set_dir = input_dir / 'temp'
temp_set_dir.mkdir(exist_ok=True)

# Loop through all .edf files
for edf_file in input_dir.glob("*.edf"):
    # Load the raw file
    raw = mne.io.read_raw_edf(edf_file, preload=False)
    subject_id = edf_file.stem.zfill(2) # e.g., "1" -> "01"

    # Export to EEGLAB .set
    set_path  = temp_set_dir / f'{subject_id}.set'
    mne.export.export_raw(str(set_path), raw, fmt='eeglab', overwrite=True)

    # Load the .set file again using MNE
    raw_set = mne.io.read_raw_eeglab(set_path, preload=True)

    # Create a BIDSPath
    bids_path = BIDSPath(
        subject=f"{subject_id}",  # or use a real ID if available
        session="01",
        task="rest",         # adjust this if needed
        run="01",
        root=bids_root,
        datatype="eeg"
    )

    # Write to BIDS
    write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True, format='EEGLAB')

df_data = pd.read_excel(f'{input_dir}/States.xlsx')
df_data['participant_id'] = df_data['file number'].apply(lambda x: f'sub-{x:02d}')
df_data['Group'] = df_data['status'].replace({'NORMAL': 'C', 'MCI': 'D'})
df_data.iloc[:, 2:].to_csv(f'{bids_root}/participants.tsv', sep='\t', index=False)


shutil.rmtree(temp_set_dir)
os.system("spd-say 'it is done'")
