import pandas as pd
import numpy as np

files = {
    'REPLACE-BG': '../results/results_REPLACE_BG.parquet',
    'DiaTrend': '../results/results_DiaTrend.parquet',
    'T1DiabetesGranada': '../results/results_T1DiabetesGranada.parquet',
}

for name, fpath in files.items():
    df = pd.read_parquet(fpath)
    entire = df[df['Range']=='ENTIRE']
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    print(f"{'Config':<12} {'Group':<10} {'RMSE mean':>10} {'RMSE std':>10} {'MAE mean':>10} {'MAE std':>10}")
    print("-"*60)
    for prop in sorted(entire['prop_M'].unique()):
        label = f"{prop}M_{100-prop}F"
        for grp in ['male','female','combined']:
            subset = entire[(entire['prop_M']==prop)&(entire['group']==grp)]
            rmse_m = subset['rmse'].mean()
            rmse_s = subset['rmse'].std()
            mae_m  = subset['mae'].mean()
            mae_s  = subset['mae'].std()
            print(f"{label:<12} {grp:<10} {rmse_m:>10.2f} {rmse_s:>10.2f} {mae_m:>10.2f} {mae_s:>10.2f}")
