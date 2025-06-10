import os
import torch
from torch.utils.data import Dataset
import numpy as np
import mne

import pandas as pd
from datetime import datetime

import numpy as np
import pandas as pd
from datetime import datetime
import mne

def parse_time_str(t_str):
    """Parses time in hh.mm.ss format to seconds since midnight."""
    t_str = t_str.replace(":", ".")
    t_obj = datetime.strptime(t_str, "%H.%M.%S")
    return t_obj.hour * 3600 + t_obj.minute * 60 + t_obj.second

def read_remlogic_txt_annotations(file_path, meas_date)  -> mne.Annotations:
    """
    Parses a REMlogic exported .txt annotation file and returns MNE-compatible annotations.
    
    Parameters:
        file_path (str): Path to the REMlogic .txt file.
    
    Returns:
        annotations (mne.Annotations): MNE annotations object containing CAP events.
        hypnogram (pd.DataFrame): Sleep stage annotations.
    """

    meas_date = meas_date.replace(tzinfo=None)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header_found = False
    data_lines = []
    cols = []
    for line in lines:
        if line.strip().startswith("Sleep Stage"):
            cols = line.split('\t')
            header_found = True
            continue
        if header_found:
            if line.strip():  # skip blank lines
                parts = line.split('\t')
                if len(parts) < len(cols):
                    continue
                data_lines.append(line.strip())

    df = pd.DataFrame([l.split('\t') for l in data_lines],
                      columns=cols)

    # Normalize column types
    df['Seconds'] = df['Time [hh:mm:ss]'].apply(parse_time_str)
    df['Duration[s]'] = pd.to_numeric(df['Duration[s]'], errors='coerce')
    
    # Detect midnight rollover and fix wraparound
    seconds = df['Seconds'].to_numpy()
    adjusted_seconds = []
    offset = 0
    for i in range(len(seconds)):
        if i > 0 and seconds[i] < seconds[i - 1]:
            offset += 86400  # Add 24h on rollover
        adjusted_seconds.append(seconds[i] + offset)

    first_event_datetime = datetime.combine(meas_date.date(), datetime.min.time()) + pd.to_timedelta(adjusted_seconds, unit='s')
    df['AbsoluteTime'] = first_event_datetime
    df['RelativeToMeas'] = (df['AbsoluteTime'] - meas_date).dt.total_seconds()

    
    cap_df = df[df['Event'].str.contains(r'^(MCAP-A|SLEEP-)', na=False)].copy()
    cap_onsets = cap_df['RelativeToMeas'].values.astype(float)
    cap_durations = cap_df['Duration[s]'].values.astype(float)
    cap_descriptions = cap_df['Event'].values

    # Create MNE Annotations
    annotations = mne.Annotations(onset=cap_onsets,
                                   duration=cap_durations,
                                   description=cap_descriptions)

    return annotations


class CapSleepDatasetHRV(Dataset):
    def __init__(self, edf_filepaths, window_size=30, sfreq=256, transform=None):
        self.edf_filepaths = edf_filepaths
        self.window_size = window_size
        self.sfreq = sfreq
        self.transform = transform
        self.windows = self._index_all_windows()
        self.label_map = {
            "MCAP-A1": 0,
            "MCAP-A2": 1,
            "MCAP-A3": 2,
            "SLEEP-MT": 3,
            "SLEEP-REM": 4,
            "SLEEP-S0": 5,
            "SLEEP-S1": 6,
            "SLEEP-S2": 7,
            "SLEEP-S3": 8,
            "SLEEP-S4": 9,
            "SLEEP-UNSCORED": 10,
        }


    def _index_all_windows(self):
        all_windows = []
        for filepath in self.edf_filepaths:
            ann_path = filepath.replace("edf", "txt")
            if not os.path.isfile(ann_path):
                continue

            raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
            ecg_ch = self._get_ecg_channel_name(raw)
            if not ecg_ch:
                continue
            duration = raw.times[-1]
            num_windows = int(duration) // self.window_size
            all_windows.extend([(filepath, i, ecg_ch, ann_path) for i in range(num_windows)])
        return all_windows

    def _get_ecg_channel_name(self, raw):
        for ch in raw.info['ch_names']:
            if 'ecg' in ch.lower() or 'ekg' in ch.lower():
                return ch
        return None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        filepath, window_idx, ecg_ch, ann_path = self.windows[idx]

        # Read EDF + annotations
        raw = mne.io.read_raw_edf(filepath, preload=True, include=[ecg_ch], verbose=False)
        annotations = read_remlogic_txt_annotations(ann_path, raw.info['meas_date'])
        raw.set_annotations(annotations)

        # Resample if needed
        if raw.info['sfreq'] != self.sfreq:
            raw.resample(self.sfreq, npad='auto')

        start_sec = window_idx * self.window_size
        stop_sec = start_sec + self.window_size

        # Crop to window
        raw_window = raw.copy().crop(tmin=start_sec, tmax=stop_sec, include_tmax=False)

        # Detect R-peaks
        try:
            ecg_channel_index = 0  # Only one channel is loaded
            _, ecg_events = mne.preprocessing.find_ecg_events(raw_window, ch_name=ecg_ch, verbose=False)
            beat_times = raw_window.times[ecg_events[:, 0]]
            rr_intervals = np.diff(beat_times)
        except Exception:
            # If detection fails, return zeros
            rr_intervals = np.zeros(1)

        # Pad or truncate RR intervals to fixed length
        max_rr_count = 20
        if len(rr_intervals) < max_rr_count:
            rr_intervals = np.pad(rr_intervals, (0, max_rr_count - len(rr_intervals)), mode='constant')
        else:
            rr_intervals = rr_intervals[:max_rr_count]

        # --------- LABEL EXTRACTION ---------
        overlapping = annotations.crop(tmin=start_sec, tmax=stop_sec)
        cap_labels = [d for d in overlapping.description]
        if cap_labels:
            label = self.label_map[cap_labels[0]]
        else:
            label = self.label_map["SLEEP-UNSCORED"]

        return torch.tensor(rr_intervals, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


