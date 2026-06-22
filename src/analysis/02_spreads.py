import pandas as pd
import numpy as np

files = {
    'REPLACE-BG': '../results/results_REPLACE_BG.parquet',
    'DiaTrend': '../results/results_DiaTrend.parquet',
    'T1DiabetesGranada': '../results/results_T1DiabetesGranada.parquet',
}

print(f"{'Dataset':<20} {'Group':<10} {'Metric':<8} {'Min':>8} {'Max':>8} {'Spread':>8}")
print("-"*70)
for name, fpath in files.items():
    df = pd.read_parquet(fpath)
    entire = df[df['Range']=='ENTIRE']
    for grp in ['male','female']:
        for metric in ['rmse','mae']:
            vals = entire[entire['group']==grp].groupby('prop_M')[metric].mean()
            print(f"{name:<20} {grp:<10} {metric:<8} {vals.min():>8.2f} {vals.max():>8.2f} {vals.max()-vals.min():>8.2f}")
