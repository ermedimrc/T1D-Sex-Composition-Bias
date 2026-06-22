# -*- coding: utf-8 -*-
"""
3_experiment.py
Sex-proportion bias experiment for blood glucose prediction.
TFG — Mehdi Iabouten | Universidad de Granada 2026

and Test_predictions.py 
"""

# ── Imports ────────────────────────────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import Dense, LSTM, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras

import numpy as np
import tensorflow as tf
import random
import os
import pandas as pd
import time
import datetime
import json
import math
import socket
import platform

# ── Reproducibility  ────────────────────────────────────────────
np.random.seed(50)
tf.random.set_seed(50)
random.seed(50)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# ── Config ─────────────────────────────────────────────────────────────────────
batch_size        = 4096
patience          = 10
max_epoch         = 500
dropout           = 0.1
recurrent_dropout = 0.0
history_length    = 8
#dataset_name      = 'REPLACE_BG'   
#dataset_name      = 'DiaTrend'
dataset_name      = 'T1DiabetesGranada'

windows_path = 'data/folds/'
sex_files = {
    'REPLACE_BG':        ('parquet', 'data/sex/sex_REPLACE_BG.parquet'),
    'DiaTrend':          ('parquet', 'data/sex/sex_DiaTrend.parquet'),
    'T1DiabetesGranada': ('csv',     'data/sex/sex_T1DiabetesGranada.csv'),
}
results_dir = 'results/'
k_folds     = 5

PROPORTIONS = [
    (100,  0), (90, 10), (80, 20), (70, 30), (60, 40),
    (50, 50),
    (40, 60), (30, 70), (20, 80), (10, 90), (0, 100),
]

#k_folds     = 1
#PROPORTIONS = [(50, 50)]

used_columns = [f'x{i}' for i in range(history_length)] + ['y']
# ── End Config ─────────────────────────────────────────────────────────────────

# ── Clarke Error Grid (Test_predictions.py 
def clarke_error_grid(ref_values, pred_values):
    ref_values  = np.asarray(ref_values,  dtype=np.float32).clip(0, 500)
    pred_values = np.asarray(pred_values, dtype=np.float32).clip(0, 500)

    zone = [0, 0, 0, 0, 0]  # A, B, C, D, E

    for i in range(len(ref_values)):
        r, p = ref_values[i], pred_values[i]

        # Zone A
        if ((r < 70 and p < 70)
                or (p <= 1.2 * r and p >= 0.8 * r)):
            zone[0] += 1

        # Zone E
        elif ((r > 180 and p < 70)
              or (r < 70 and p > 180)):
            zone[4] += 1

        # Zone C
        elif (((r >= 70 and r <= 290) and p >= r + 110)
              or ((r >= 130 and r <= 180) and (p <= (7 / 5) * r - 182))):
            zone[2] += 1

        # Zone D
        elif ((r > 240 and (p >= 70 and p <= 180))
              or (r <= 175 / 3 and p <= 180 and p >= 70)
              or ((r >= 175 / 3 and r < 70) and p >= (6 / 5) * r)):
            zone[3] += 1

        # Zone B
        else:
            zone[1] += 1

    n = len(ref_values)
    zone_pct = [round(z / n * 100, 2) for z in zone]
    return zone, zone_pct

def metrics_by_range(df_result):
    """Compute Clarke zones and error metrics per glycaemic range.
    Same ranges as test_by_range() in Test_predictions.py."""
    ranges = {
        'ENTIRE': df_result,
        'TBR_2':  df_result[df_result['y_test'] < 54],
        'TBR_1':  df_result[(df_result['y_test'] >= 54) & (df_result['y_test'] < 70)],
        'TIR':    df_result[(df_result['y_test'] >= 70) & (df_result['y_test'] < 181)],
        'TAR_1':  df_result[(df_result['y_test'] >= 181) & (df_result['y_test'] < 251)],
        'TAR_2':  df_result[df_result['y_test'] >= 251],
    }
    rows = []
    for range_name, df_r in ranges.items():
        if len(df_r) == 0:
            continue
        ref  = df_r['y_test'].values
        pred = df_r['y_predict'].values
        zone, zone_pct = clarke_error_grid(ref, pred)
        mae  = mean_absolute_error(ref, pred)
        rmse = np.sqrt(mean_squared_error(ref, pred))
        rows.append({
            'Range': range_name,
            'A':     zone_pct[0],
            'B':     zone_pct[1],
            'C':     zone_pct[2],
            'D':     zone_pct[3],
            'E':     zone_pct[4],
            'A + B': zone_pct[0] + zone_pct[1],
            'MAE':   round(mae, 4),
            'RMSE':  round(rmse, 4),
            'N':     len(df_r),
        })
    return pd.DataFrame(rows)

# ── Model (LSTMModel.build() 
def build_lstm_model():
    i = Input(shape=(history_length, 1))
    x = LSTM(128, dropout=dropout, recurrent_dropout=recurrent_dropout)(i)
    x = Dense(1, activation=None)(x)
    model = Model(inputs=[i], outputs=[x])
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.RootMeanSquaredError()],
    )
    return model

# ── Callbacks (make_callbacks() 
def make_callbacks(filepath_prefix):
    return [
        ModelCheckpoint(filepath=f'{filepath_prefix}.weights.h5',
                        monitor='val_loss', mode='min',
                        save_best_only=True, save_weights_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', mode='min',
                      patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.TensorBoard(log_dir=f'{filepath_prefix}_logs',
                                    histogram_freq=0, write_graph=True,
                                    write_images=False, update_freq='epoch',
                                    profile_batch=0),
        keras.callbacks.CSVLogger(filename=f'{filepath_prefix}_history.csv',
                                  separator=',', append=False),
    ]

# ── Best epoch JSON (write_fold_json_and_md 
def write_best_json(save_prefix, fold_idx, label, dataset_name,
                    train_time, data_counts):
    hist_csv = f'{save_prefix}_history.csv'
    best_json = f'{save_prefix}.best.json'
    weights   = f'{save_prefix}.weights.h5'

    try:
        df_hist  = pd.read_csv(hist_csv)
        best_idx = int(df_hist['val_loss'].idxmin()
                       if 'val_loss' in df_hist.columns
                       else df_hist['loss'].idxmin())
        row = df_hist.iloc[best_idx].to_dict()
        payload = {
            'experiment': {
                'dataset': dataset_name, 'fold': fold_idx, 'label': label,
                'algorithm': 'LSTM', 'history_length': history_length,
                'started_at': datetime.datetime.now().isoformat(),
            },
            'data': {'counts': data_counts},
            'timing': {
                'train_seconds': round(train_time, 3),
                'train_elapsed': str(datetime.timedelta(seconds=train_time)),
            },
            'best_epoch': {
                'monitor': 'val_loss', 'epoch_idx': best_idx,
                'epoch_count': int(df_hist.shape[0]), 'metrics': row,
            },
            'artifacts': {
                'weights_path': weights,
                'history_csv_path': hist_csv,
            },
            'system': {
                'hostname': socket.gethostname(),
                'os': f'{platform.system()} {platform.release()}',
                'python': platform.python_version(),
                'tensorflow': tf.__version__,
            },
        }
        with open(best_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f'  WARNING: could not write best.json for {label}: {e}')

# ── Train function (train() 
def train(x_train, y_train, x_val, y_val, model, save_prefix=''):
    x_train = x_train.reshape(x_train.shape[0], history_length, 1)
    x_val   = x_val.reshape(x_val.shape[0], history_length, 1)
    hist = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        epochs=max_epoch,
        callbacks=make_callbacks(save_prefix),
    )
    return hist, model

# ── Sex-proportion sampling ────────────────────────────────────────────────────
def balanced_sample(patients, patient_windows_dict, n_select, seed=50, block_size=10):
    rng = random.Random(seed)
    n_select = min(n_select, len(patients))
    if n_select <= 0:
        return []
    if n_select >= len(patients):
        return list(patients)

    sorted_pats = sorted(patients, key=lambda p: patient_windows_dict[p], reverse=True)
    n_total = len(sorted_pats)
    ratio   = n_select / n_total
    selected = []
    accum = 0.0

    for b in range(math.ceil(n_total / block_size)):
        block  = sorted_pats[b * block_size: (b + 1) * block_size]
        accum += ratio * len(block)
        n_pick = min(round(accum), len(block))
        if n_pick > 0:
            chosen = rng.sample(block, n_pick)
            selected.extend(chosen)
            accum -= n_pick

    pool = [p for p in sorted_pats if p not in set(selected)]
    while len(selected) < n_select and pool:
        pick = rng.choice(pool)
        selected.append(pick)
        pool.remove(pick)

    return selected[:n_select]

def sex_proportion_sample(train_M, train_F, patient_windows_dict, r_M, r_F):
    if r_F == 0:
        return list(train_M), []
    if r_M == 0:
        return [], list(train_F)

    nM, nF    = len(train_M), len(train_F)
    n_prime_F = math.floor(nM * r_F / r_M)

    if n_prime_F <= nF:
        sel_M = list(train_M)
        sel_F = balanced_sample(train_F, patient_windows_dict, n_prime_F)
    else:
        n_prime_M = math.floor(nF * r_M / r_F)
        sel_M = balanced_sample(train_M, patient_windows_dict, n_prime_M)
        sel_F = list(train_F)

    return sel_M, sel_F

def combined_test_sample(test_M_pids, test_F_pids, patient_windows_dict, r_M, r_F):
    """Build combined test set with same M/F proportion as training."""
    if r_F == 0:
        return list(test_M_pids), []
    if r_M == 0:
        return [], list(test_F_pids)

    nM, nF    = len(test_M_pids), len(test_F_pids)
    n_prime_F = math.floor(nM * r_F / r_M)

    if n_prime_F <= nF:
        sel_M = list(test_M_pids)
        sel_F = balanced_sample(test_F_pids, patient_windows_dict, n_prime_F)
    else:
        n_prime_M = math.floor(nF * r_M / r_F)
        sel_M = balanced_sample(test_M_pids, patient_windows_dict, n_prime_M)
        sel_F = list(test_F_pids)

    return sel_M, sel_F

# ── Main ───────────────────────────────────────────────────────────────────────
os.makedirs(results_dir, exist_ok=True)

# Load windows with fold tags
data_file_path = os.path.join(windows_path, f'windows_folds_{dataset_name}.parquet')
print(f'Loading: {data_file_path}')
df = pd.read_parquet(data_file_path)

# Load sex map
fmt, sex_path = sex_files[dataset_name]
df_sex = pd.read_csv(sex_path) if fmt == 'csv' else pd.read_parquet(sex_path)
sex_map = dict(zip(df_sex['Patient_ID'].astype(str), df_sex['Sex'].astype(str)))
df['sex'] = df['patient_id'].astype(str).map(sex_map)

# Load folds (with train_M, train_F, test_M, test_F)
folds_path = os.path.join(windows_path, f'folds_{dataset_name}.json')
with open(folds_path) as f:
    folds = json.load(f)

# Patient windows count — string keys to match JSON fold format
patient_windows_dict = {str(k): int(v) for k, v in df.groupby('patient_id').size().items()}

all_results = []

for fold_idx in range(k_folds):
    fold = folds[fold_idx]
    current_fold = fold_idx + 1

    tf.keras.backend.clear_session()
    np.random.seed(50)
    tf.random.set_seed(50)
    random.seed(50)

    print(f'\n{"="*60}')
    print(f'FOLD {current_fold}/{k_folds}')
    print(f'{"="*60}')

    # ── Validation set ─────────────────────────────────────────────────────────
    df_val   = df[df[f'fold_{fold_idx}'] == 'val'].reset_index(drop=True)
    x_val    = df_val[used_columns[:-1]].to_numpy()
    y_val    = df_val[['y']].to_numpy()

    # ── Test sets by sex ───────────────────────────────────────────────────────
    df_testM = df[df['patient_id'].astype(str).isin(fold['test_M'])].reset_index(drop=True)
    df_testF = df[df['patient_id'].astype(str).isin(fold['test_F'])].reset_index(drop=True)

    print(f'  Val:    {df_val["patient_id"].nunique()} patients, {len(df_val)} windows')
    print(f'  Test M: {df_testM["patient_id"].nunique()} patients, {len(df_testM)} windows')
    print(f'  Test F: {df_testF["patient_id"].nunique()} patients, {len(df_testF)} windows')

    for r_M, r_F in PROPORTIONS:
        label    = f'{r_M}M_{r_F}F'
        res_path = os.path.join(results_dir, f'{dataset_name}_fold{fold_idx}_{label}.json')

        if os.path.exists(res_path):
            print(f'  {label}: already done, loading from cache')
            with open(res_path) as f_res:
                all_results.extend(json.load(f_res))
            continue

        # ── Sex-proportion training set ────────────────────────────────────────
        sel_M, sel_F = sex_proportion_sample(
            fold['train_M'], fold['train_F'], patient_windows_dict, r_M, r_F
        )
        df_train = df[df['patient_id'].astype(str).isin(sel_M + sel_F)].reset_index(drop=True)
        x_train  = df_train[used_columns[:-1]].to_numpy()
        y_train  = df_train[['y']].to_numpy()

        print(f'\n  {label} | train: {len(sel_M)}M + {len(sel_F)}F = {len(df_train)} windows')

        # ── Train ──────────────────────────────────────────────────────────────
        model      = build_lstm_model()
        save_prefix = os.path.join(results_dir, f'{dataset_name}_fold{fold_idx}_{label}')

        print(f'  START training LSTM - Fold={current_fold} {label}')
        start_time = time.time()
        hist, model_trained = train(x_train, y_train, x_val, y_val, model, save_prefix)
        train_time = time.time() - start_time
        print(f'  END training LSTM - Fold={current_fold} {label} (Time: {datetime.timedelta(seconds=train_time)})')

        # ── Write best epoch JSON  ───────────────────────────────
        data_counts = {
            'patients': {'train': len(set(sel_M + sel_F)), 'val': int(df_val['patient_id'].nunique()),
                         'test_M': len(fold['test_M']), 'test_F': len(fold['test_F'])},
            'windows':  {'train': len(df_train), 'val': len(df_val),
                         'test_M': len(df_testM), 'test_F': len(df_testF)},
        }
        write_best_json(save_prefix, fold_idx, label, dataset_name, train_time, data_counts)

        # ── Load best weights ──────────────────────────────────────────────────
        model_best = build_lstm_model()
        model_best.load_weights(f'{save_prefix}.weights.h5')

        # ── Corrected combined test (same proportion as training) ──────────────
        comb_M_pids, comb_F_pids = combined_test_sample(
            fold['test_M'], fold['test_F'], patient_windows_dict, r_M, r_F
        )
        df_test_combined = df[
            df['patient_id'].astype(str).isin(comb_M_pids + comb_F_pids)
        ].reset_index(drop=True)

        # ── Predict and evaluate ───────────────────────────────────────────────
        fold_results = []
        for df_grp, group_name in [
            (df_testM,         'male'),
            (df_testF,         'female'),
            (df_test_combined, 'combined'),
        ]:
            if len(df_grp) == 0:
                continue

            x_test = df_grp[used_columns[:-1]].to_numpy()
            y_test = df_grp[['y']].to_numpy()
            x_test_reshaped = x_test.reshape(x_test.shape[0], history_length, 1)
            y_pred = model_best.predict(x_test_reshaped, batch_size=batch_size, verbose=0)

            df_res    = pd.DataFrame({'y_test': y_test.ravel(), 'y_predict': y_pred.ravel()})
            df_metrics = metrics_by_range(df_res)

            for _, row in df_metrics.iterrows():
                fold_results.append({
                    'fold':         fold_idx,
                    'prop_M':       r_M,
                    'prop_F':       r_F,
                    'label':        label,
                    'group':        group_name,
                    'dataset':      dataset_name,
                    'Range':        row['Range'],
                    'mae':          row['MAE'],
                    'rmse':         row['RMSE'],
                    'zone_A':       row['A'],
                    'zone_B':       row['B'],
                    'zone_C':       row['C'],
                    'zone_D':       row['D'],
                    'zone_E':       row['E'],
                    'zone_AB':      row['A + B'],
                    'n':            row['N'],
                    'n_train_M':    len(sel_M),
                    'n_train_F':    len(sel_F),
                    'train_time_s': round(train_time, 2),
                })

        # ── Save fold results ──────────────────────────────────────────────────
        with open(res_path, 'w') as f_res:
            json.dump(fold_results, f_res)
        all_results.extend(fold_results)

        # Print summary
        m = next((r for r in fold_results if r['group'] == 'male'   and r['Range'] == 'ENTIRE'), None)
        f = next((r for r in fold_results if r['group'] == 'female' and r['Range'] == 'ENTIRE'), None)
        if m and f:
            print(f'  MAE male={m["mae"]:.2f}  female={f["mae"]:.2f}  (train {train_time:.0f}s)')

# ── Save aggregated results ────────────────────────────────────────────────────
df_all   = pd.DataFrame(all_results)
out_path = os.path.join(results_dir, f'results_{dataset_name}.parquet')
df_all.to_parquet(out_path, index=False)
print(f'\n✓ Done. Saved: {out_path}  shape={df_all.shape}')
