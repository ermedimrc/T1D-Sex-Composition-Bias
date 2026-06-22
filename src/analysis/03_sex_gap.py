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
    gaps_r, gaps_m = [], []
    print(f"{'Config':<12} {'gap_RMSE':>10} {'gap_MAE':>10}")
    print("-"*35)
    for prop in sorted(entire['prop_M'].unique()):
        mr = entire[(entire['prop_M']==prop)&(entire['group']=='male')]['rmse'].mean()
        fr = entire[(entire['prop_M']==prop)&(entire['group']=='female')]['rmse'].mean()
        mm = entire[(entire['prop_M']==prop)&(entire['group']=='male')]['mae'].mean()
        fm = entire[(entire['prop_M']==prop)&(entire['group']=='female')]['mae'].mean()
        gaps_r.append(fr-mr)
        gaps_m.append(fm-mm)
        print(f"{prop}M_{100-prop}F    {fr-mr:>10.4f} {fm-mm:>10.4f}")
    print("-"*35)
    print(f"{'mean':<12} {np.mean(gaps_r):>10.4f} {np.mean(gaps_m):>10.4f}")
    print(f"{'std':<12} {np.std(gaps_r):>10.4f} {np.std(gaps_m):>10.4f}")
    print(f"{'band(max-min)':<12} {max(gaps_r)-min(gaps_r):>10.4f} {max(gaps_m)-min(gaps_m):>10.4f}")
