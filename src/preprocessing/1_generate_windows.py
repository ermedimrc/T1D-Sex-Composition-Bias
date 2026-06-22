# -*- coding: utf-8 -*-
"""
1_generate_windows.py
Generates sliding-window datasets from filtered CGM parquet files.
"""

import numpy as np
import pandas as pd
import os

# ── Config ─────────────────────────────────────────────────────────────────────
HISTORY_LENGTH             = 8       # 8 × 15 min = 2 hours
HORIZON                    = 4       # 4 × 15 min = 1 hour ahead
DEDUPLICATE_INTRA_PATIENT  = True

DATASETS = {
    'REPLACE_BG': 'data/raw/Glucose_measurements_REPLACE-BG_FILTERED_2026-03-27.parquet',
    'DiaTrend':   'data/raw/Glucose_measurements_DiaTrend_FILTERED_2026-03-27.parquet',
    'T1DiabetesGranada': 'data/raw/Glucose_measurements_T1DiabetesGranada_FILTERED_2026-03-27.parquet',
}

OUTPUT_DIR = 'data/windows/'
# ── End Config ─────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_windows_one_step_walk_forward(
    patient_dict: dict,
    date_dict: dict,
    time_dict: dict,
    history_length: int,
    horizon: int,
    deduplicate_intra_patient: bool,
) -> pd.DataFrame:

    dfs = []

    for patient_id, patient_series in patient_dict.items():
        x       = np.lib.stride_tricks.sliding_window_view(patient_series[:-horizon], history_length)
        x_date  = np.lib.stride_tricks.sliding_window_view(date_dict[patient_id][:-horizon], history_length)
        x_time  = np.lib.stride_tricks.sliding_window_view(time_dict[patient_id][:-horizon], history_length)
        y_all   = np.lib.stride_tricks.sliding_window_view(patient_series[history_length:], horizon)

        nan_rows_x = np.isnan(x).any(axis=1)
        nan_rows_y = np.isnan(y_all).any(axis=1)
        mask = ~(nan_rows_x | nan_rows_y)

        x      = x[mask]
        x_date = x_date[mask]
        x_time = x_time[mask]
        y_all  = y_all[mask]

        if len(x) == 0:
            continue

        df_x      = pd.DataFrame(x).round(1)
        df_x_date = pd.DataFrame(x_date[:, -1])
        df_x_time = pd.DataFrame(x_time[:, -1])
        df_y      = pd.DataFrame(y_all[:, -1]).round(1)

        df_set = pd.concat([df_x, df_y, df_x_date, df_x_time], axis=1)
        df_set.columns = (
            [f'x{i}' for i in range(history_length)]
            + ['y']
            + [f'x_date_{history_length - 1}']
            + [f'x_time_{history_length - 1}']
        )
        df_set['patient_id'] = patient_id
        dfs.append(df_set)

    df_all = pd.concat(dfs, ignore_index=True)

    if deduplicate_intra_patient:
        print(f'  Deduplicating... {len(df_all):,} rows before')
        dup_cols = [f'x{i}' for i in range(history_length)] + ['y', 'patient_id']
        df_all['_is_dup'] = df_all.duplicated(subset=dup_cols, keep='first')
        df_all = df_all[~df_all['_is_dup']].drop(columns=['_is_dup'])
        df_all.reset_index(drop=True, inplace=True)
        print(f'  Deduplicating... {len(df_all):,} rows after')

    return df_all

for dataset_name, data_path in DATASETS.items():
    print(f'\n{"="*60}')
    print(f'Processing: {dataset_name}')
    print(f'{"="*60}')

    out_path = os.path.join(OUTPUT_DIR, f'windows_{dataset_name}_H{HORIZON}.parquet')
    if os.path.exists(out_path):
        print(f'  Already exists, skipping: {out_path}')
        continue

    df_data = pd.read_parquet(data_path)
    df_valid = df_data[df_data['15min'].notna()].copy()
    df_valid = df_valid.sort_values(['patient_id', '15min']).reset_index(drop=True)

    print(f'  Valid rows: {len(df_valid):,} | Patients: {df_valid["patient_id"].nunique()}')

    patient_dict = {
        pid: grp['measurement'].to_numpy(dtype=float, na_value=np.nan)
        for pid, grp in df_valid.groupby('patient_id')
    }
    date_dict = {
        pid: grp['15min'].dt.date.to_numpy()
        for pid, grp in df_valid.groupby('patient_id')
    }
    time_dict = {
        pid: grp['15min'].dt.time.to_numpy()
        for pid, grp in df_valid.groupby('patient_id')
    }

    df_windows = get_windows_one_step_walk_forward(
        patient_dict=patient_dict,
        date_dict=date_dict,
        time_dict=time_dict,
        history_length=HISTORY_LENGTH,
        horizon=HORIZON,
        deduplicate_intra_patient=DEDUPLICATE_INTRA_PATIENT,
    )

    df_windows.to_parquet(out_path, index=False)
    print(f'  Saved: {out_path} ({len(df_windows):,} windows)')

print('\nDone.')
