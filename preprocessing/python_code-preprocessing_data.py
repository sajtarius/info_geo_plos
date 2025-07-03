# @title preprocessing the data and save them in [derivatives/preprocessing_pipeline] folder.
import mne
import os
import asrpy

from scipy.signal import detrend
from mne_icalabel import label_components
from pathlib import Path
from bids import BIDSLayout
from bids.tests import get_test_data_path
from mne_bids import BIDSPath, write_raw_bids, make_report
from mne_bids.utils import _write_json
from mne_bids.sidecar_updates import update_sidecar_json
from tqdm import tqdm


# --- CONFIGURATION ---
bids_root = Path("/home/hengjie/Downloads/alzheimer_data-nature/bids_dataset")  # root of your BIDS dataset
deriv_root = bids_root / "derivatives" / "preprocessing_pipeline"
deriv_root.mkdir(parents=True, exist_ok=True)

data_path = os.path.join(get_test_data_path(), f'{bids_root}')
layout = BIDSLayout(data_path)
all_file = layout.get(return_type='filename', extension='.set')

for i in tqdm(all_file, position=0):
    # subject/session/task/run info
    subject = i.split('/')[-1].split('_')[0].split('-')[-1]
    session = i.split('/')[-1].split('_')[1].split('-')[-1]
    task = i.split('/')[-1].split('_')[2].split('-')[-1]
    run = i.split('/')[-1].split('_')[3].split('-')[-1]

    # Load the data and conducting the preprocessing
    temp_data = mne.io.read_raw_eeglab(i, preload=True)
    temp_data.resample(sfreq=500) # resampling to 500Hz to match the previous data set sampling frequency

    # Bandpass filtering; the range is set from 1 to 100Hz for the [automatic labelling ICA] later
    temp_data.filter(l_freq=1, h_freq=100,)# method='iir', iir_params=dict(order=4, ftype='butter'))
    temp_data.notch_filter(freqs=50) # notch filtering to filter 50Hz of the power source frequency.
    temp_data._data = detrend(temp_data._data, axis=1, type='constant') # detrending the data based on [constant] trend

    # Artifact Subspace Reconstruction
    asr = asrpy.ASR(sfreq=temp_data.info['sfreq'], cutoff=2.5, win_len=0.5) # [cutoff] set to 2.5 is the EXTREME cutoff that still able to retain the EEG signals' information.
    asr.fit(temp_data)
    temp_data_asr = asr.transform(temp_data)


    # [common average referencing] this is needed for the [automatic labelling ICA] later; it MUST be done AFTER [Bandpass Filtering] and [Artifact Subspace Reconstruction].
    # the [common average referencing] is not done because it will amplify the artifacts!!!
    #temp_data_asr.set_eeg_reference(ref_channels='average')


    # Independent Component Analysis
    temp_data_asr.set_montage('standard_1020') #setting the montage configuration for ICA
    ica = mne.preprocessing.ICA(n_components=18, method='infomax', fit_params=dict(extended=True), random_state=34) #the method of ['infomax'] and fit_params of [dict(extended=True)] are used as suggested by the mne_icalabel
    ica.fit(temp_data_asr)

    # Independent Component Analysis; automatic labelling
    ica_labels = label_components(temp_data_asr, ica, 'iclabel')
    ica_labels_name = ica_labels['labels']
    ica_labels_prob = ica_labels['y_pred_proba']
    print(f'ICA labels name: {ica_labels_name}')
    print(f'ICA labels prob: {ica_labels_prob}')

    # Independent Component Analysis; indices to be excluded
    exclude_idx = [idx for idx, label in enumerate(ica_labels_name) if label not in ['brain', 'other']]
    print(f'ICA indices to be excluded: {exclude_idx}')

    raw_preproc = ica.apply(temp_data_asr, exclude=exclude_idx)


    # Create the BIDSPath
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        suffix="eeg",
        extension=".set",
        root=deriv_root,
        datatype="eeg",
        #desc="cleaned"
    )

    # Write the preprocessed file to the derivatives folder
    write_raw_bids(
        raw=raw_preproc,
        bids_path=bids_path,
        format="EEGLAB",  # or 'EDF', 'BrainVision', 'EEGLAB', etc.
        overwrite=True,
        allow_preload=True
    )
    '''
    # Update dataset_description.json
    dataset_description_path = deriv_root / "dataset_description.json"
    description_dict = {
        "Name": "MyEEGPreprocessing",
        "BIDSVersion": "1.8.0",
        "PipelineDescription": {
            "Name": "EEG-Cleaning-Pipeline",
            "Version": "1.0",
            "CodeURL": "https://github.com/myusername/eeg-cleaning"
        }
    }
    _write_json(dataset_description_path, description_dict)
    '''

    print(f"Preprocessed data saved to: {bids_path.directory}")

os.system("spd-say 'it is done'")
