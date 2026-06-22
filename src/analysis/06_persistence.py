# -*- coding: utf-8 -*-
"""
06_persistence_from_data.py
Calcula el RMSE del modelo de persistencia (lag-4 = 1 hora) usando
los mismos windows_folds y folds JSON que el experimento principal.
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# ── Config ────────────────────────────────────────────────────────────────────
history_length = 8
windows_path   = '../data/folds/'
sex_files = {
    'REPLACE_BG':        ('parquet', '../data/sex/sex_REPLACE_BG.parquet'),
    'DiaTrend':          ('parquet', '../data/sex/sex_DiaTrend.parquet'),
    'T1DiabetesGranada': ('csv',     '../data/sex/sex_T1DiabetesGranada.csv'),
}
k_folds = 5
DATASETS = ['REPLACE_BG', 'DiaTrend', 'T1DiabetesGranada']
# ──────────────────────────────────────────────────────────────────────────────

# x7 es el valor mas reciente (t), y es el valor en t+1h
# Persistencia = predecir y_pred = x7
LAST_X = f'x{history_length - 1}'  # = 'x7'

print(f"Usando columna de persistencia: {LAST_X}")
print(f"{'Dataset':<20} {'Group':<10} {'Fold':<6} {'Persistence RMSE':>18}")
print("-"*60)

summary = {}

for dataset_name in DATASETS:
    df = pd.read_parquet(f'{windows_path}windows_folds_{dataset_name}.parquet')

    fmt, sex_path = sex_files[dataset_name]
    df_sex = pd.read_csv(sex_path) if fmt == 'csv' else pd.read_parquet(sex_path)
    sex_map = dict(zip(df_sex['Patient_ID'].astype(str), df_sex['Sex'].astype(str)))
    df['sex'] = df['patient_id'].astype(str).map(sex_map)

    with open(f'{windows_path}folds_{dataset_name}.json') as f:
        folds = json.load(f)

    fold_results = {'male': [], 'female': []}

    for fold_idx in range(k_folds):
        fold = folds[fold_idx]
        df_testM = df[df['patient_id'].astype(str).isin(fold['test_M'])]
        df_testF = df[df['patient_id'].astype(str).isin(fold['test_F'])]

        for df_grp, grp_name in [(df_testM, 'male'), (df_testF, 'female')]:
            if len(df_grp) == 0:
                continue
            y_true = df_grp['y'].values
            y_pred = df_grp[LAST_X].values  # x7 = ultimo valor conocido
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            fold_results[grp_name].append(rmse)
            print(f"{dataset_name:<20} {grp_name:<10} {fold_idx:<6} {rmse:>18.4f}")

    male_avg   = np.mean(fold_results['male'])
    female_avg = np.mean(fold_results['female'])
    gap        = female_avg - male_avg
    summary[dataset_name] = {'male': male_avg, 'female': female_avg, 'gap': gap}
    print()

print("\n" + "="*60)
print("SUMMARY (persistence RMSE averaged over 5 folds)")
print("="*60)
print(f"{'Dataset':<20} {'Male':>10} {'Female':>10} {'Gap':>10}")
print("-"*55)
for name, v in summary.items():
    print(f"{name:<20} {v['male']:>10.2f} {v['female']:>10.2f} {v['gap']:>10.2f}")