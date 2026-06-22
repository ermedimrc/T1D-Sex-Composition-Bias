# -*- coding: utf-8 -*-
"""
2_folds_creation.py
Creates 5-fold cross-validation splits stratified by sex.
"""

import pandas as pd
import os
import random
from collections import Counter
from typing import Dict, List

Fold = Dict[str, List[str]]

# ── Config ─────────────────────────────────────────────────────────────────────
DATASETS = {
    'REPLACE_BG':        'data/windows/windows_REPLACE_BG_H4.parquet',
    'DiaTrend':          'data/windows/windows_DiaTrend_H4.parquet',
    'T1DiabetesGranada': 'data/windows/windows_T1DiabetesGranada_H4.parquet',
}

SEX_FILES = {
    'REPLACE_BG':        ('parquet', 'data/sex/sex_REPLACE_BG.parquet'),
    'DiaTrend':          ('parquet', 'data/sex/sex_DiaTrend.parquet'),
    'T1DiabetesGranada': ('csv',     'data/sex/sex_T1DiabetesGranada.csv'),
}

OUTPUT_DIR = 'data/folds/'
K_FOLDS    = 5
SEED       = 42
# ── End Config ─────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

def make_folds_greedy_splits_balanced(
    patient_windows: Dict[str, int],
    k: int = 5,
    seed: int = 42,
) -> List[Fold]:
    """Balanced k-fold patient-wise split ."""

    rng = random.Random(seed)

    sorted_patients = sorted(
        patient_windows.keys(),
        key=lambda pid: patient_windows[pid],
        reverse=True,
    )

    # ── Test assignment ────────────────────────────────────────────────────────
    test_folds: List[List[str]] = [[] for _ in range(k)]
    fold_loads: List[int] = [0] * k

    idx = 0
    while idx + k <= len(sorted_patients):
        block = sorted_patients[idx: idx + k]
        block.sort(key=lambda pid: patient_windows[pid], reverse=True)
        rng.shuffle(block)
        for fold_idx, pid in enumerate(block):
            test_folds[fold_idx].append(pid)
            fold_loads[fold_idx] += patient_windows[pid]
        idx += k

    for pid in sorted_patients[idx:]:
        lightest = fold_loads.index(min(fold_loads))
        test_folds[lightest].append(pid)
        fold_loads[lightest] += patient_windows[pid]

    # ── Train/Val assignment ───────────────────────────────────────────────────
    folds: List[Fold] = []
    block_tv = 8  # 7 train + 1 val

    for fold_idx in range(k):
        test_ids   = set(test_folds[fold_idx])
        remaining  = [p for p in sorted_patients if p not in test_ids]

        train_ids, val_ids = [], []
        ptr = 0

        while ptr + block_tv <= len(remaining):
            block = remaining[ptr: ptr + block_tv]
            rng.shuffle(block)
            val_ids.append(block[0])
            train_ids.extend(block[1:])
            ptr += block_tv

        tail = remaining[ptr:]
        if tail:
            rng.shuffle(tail)
            val_ids.append(tail[0])
            train_ids.extend(tail[1:])

        folds.append({
            'train': sorted(train_ids),
            'val':   sorted(val_ids),
            'test':  sorted(test_folds[fold_idx]),
        })

    return folds

def make_folds_stratified_by_sex(
    patient_windows: Dict[str, int],
    sex_map: Dict[str, str],
    k: int = 5,
    seed: int = 42,
) -> List[Fold]:
    """
    Builds folds stratified by sex: male and female patients are folded
    independently and then merged. Each fold contains:
      train_M, train_F, val_M, val_F, test_M, test_F
    for use in sex-proportion sampling downstream.
    """

    males   = {p: n for p, n in patient_windows.items() if sex_map.get(p) == 'M'}
    females = {p: n for p, n in patient_windows.items() if sex_map.get(p) == 'F'}

    print(f'  Males:   {len(males)} patients')
    print(f'  Females: {len(females)} patients')

    folds_M = make_folds_greedy_splits_balanced(males,   k=k, seed=seed)
    folds_F = make_folds_greedy_splits_balanced(females, k=k, seed=seed)

    combined = []
    for i in range(k):
        combined.append({
            'train':   sorted(folds_M[i]['train'] + folds_F[i]['train']),
            'val':     sorted(folds_M[i]['val']   + folds_F[i]['val']),
            'test':    sorted(folds_M[i]['test']  + folds_F[i]['test']),
            'train_M': folds_M[i]['train'],
            'train_F': folds_F[i]['train'],
            'val_M':   folds_M[i]['val'],
            'val_F':   folds_F[i]['val'],
            'test_M':  folds_M[i]['test'],
            'test_F':  folds_F[i]['test'],
        })

    return combined

def validate_folds(folds: List[Fold], patient_windows: Dict[str, int]) -> None:
    all_patients = set(patient_windows.keys())

    for idx, fold in enumerate(folds):
        tr = set(fold['train'])
        va = set(fold['val'])
        te = set(fold['test'])

        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te), \
            f'[R1] Overlap in fold {idx}'
        assert len(tr | va | te) == len(all_patients), \
            f'[R3] Missing patients in fold {idx}'

    test_counter = Counter(pid for fold in folds for pid in fold['test'])
    duplicates = [pid for pid, cnt in test_counter.items() if cnt != 1]
    assert not duplicates, f'[R2] Patients in test != 1 time: {duplicates}'

    print('  ✓ Folds validated (R1, R2, R3)')

def tag_folds_and_save(
    df_windows: pd.DataFrame,
    folds: List[Fold],
    output_path: str,
    dataset_name: str,
) -> None:
    df_tagged = df_windows.copy()

    for idx, fold in enumerate(folds):
        label_map = {pid: 'train' for pid in fold['train']}
        label_map.update({pid: 'val'   for pid in fold['val']})
        label_map.update({pid: 'test'  for pid in fold['test']})
        df_tagged[f'fold_{idx}'] = df_tagged['patient_id'].map(label_map).astype('category')

    out_file = os.path.join(output_path, f'windows_folds_{dataset_name}.parquet')
    df_tagged.to_parquet(out_file, index=False)
    print(f'  Saved: {out_file}')

for dataset_name in DATASETS:
    print(f'\n{"="*60}')
    print(f'Dataset: {dataset_name}')
    print(f'{"="*60}')

    df_windows = pd.read_parquet(DATASETS[dataset_name])

    fmt, sex_path = SEX_FILES[dataset_name]
    df_sex = pd.read_csv(sex_path) if fmt == 'csv' else pd.read_parquet(sex_path)
    # Column names: Patient_ID, Sex
    sex_map = dict(zip(df_sex['Patient_ID'].astype(str), df_sex['Sex'].astype(str)))
    df_windows['sex'] = df_windows['patient_id'].astype(str).map(sex_map)

    patient_windows = (
        df_windows.groupby('patient_id').size().astype(int).to_dict()
    )

    folds = make_folds_stratified_by_sex(
        patient_windows=patient_windows,
        sex_map=sex_map,
        k=K_FOLDS,
        seed=SEED,
    )

    validate_folds(folds, patient_windows)

    # Print stats
    for i, fold in enumerate(folds):
        n_tr = sum(patient_windows[p] for p in fold['train'])
        n_va = sum(patient_windows[p] for p in fold['val'])
        n_te = sum(patient_windows[p] for p in fold['test'])
        total = n_tr + n_va + n_te
        print(f'  Fold {i}: train={len(fold["train"])}p ({n_tr/total*100:.1f}%) '
              f'val={len(fold["val"])}p ({n_va/total*100:.1f}%) '
              f'test={len(fold["test"])}p ({n_te/total*100:.1f}%)')

    # Save folds as JSON for use in experiment script
    import json
    folds_path = os.path.join(OUTPUT_DIR, f'folds_{dataset_name}.json')
    with open(folds_path, 'w') as f:
        json.dump(folds, f)
    print(f'  Saved: {folds_path}')

    tag_folds_and_save(df_windows, folds, OUTPUT_DIR, dataset_name)

print('\nDone.')
