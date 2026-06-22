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
    print(f"{'Config':<12} {'Male A+B':>10} {'Female A+B':>12} {'Combined A+B':>14}")
    print("-"*55)
    for prop in sorted(entire['prop_M'].unique()):
        label = f"{prop}M_{100-prop}F"
        vals = {}
        for grp in ['male','female','combined']:
            subset = entire[(entire['prop_M']==prop)&(entire['group']==grp)]
            vals[grp] = f"{subset['zone_AB'].mean():.3f}±{subset['zone_AB'].std():.3f}"
        print(f"{label:<12} {vals['male']:>10} {vals['female']:>12} {vals['combined']:>14}")
    print()
    for grp in ['male','female']:
        v = entire[entire['group']==grp].groupby('prop_M')['zone_AB'].mean()
        print(f"  {grp} spread: {v.max()-v.min():.4f} pp  (min={v.min():.3f}, max={v.max():.3f})")
