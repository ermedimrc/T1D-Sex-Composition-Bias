import pandas as pd
import numpy as np

files = {
    'REPLACE-BG': '../results/results_REPLACE_BG.parquet',
    'DiaTrend': '../results/results_DiaTrend.parquet',
    'T1DiabetesGranada': '../results/results_T1DiabetesGranada.parquet',
}

print("Symmetry: comparing 80M_20F vs 20M_80F (absolute RMSE difference)")
print(f"{'Dataset':<20} {'Group':<10} {'80M_20F':>10} {'20M_80F':>10} {'|diff|':>8}")
print("-"*65)
for name, fpath in files.items():
    df = pd.read_parquet(fpath)
    entire = df[df['Range']=='ENTIRE']
    for grp in ['male','female']:
        v80 = entire[(entire['prop_M']==80)&(entire['group']==grp)]['rmse'].mean()
        v20 = entire[(entire['prop_M']==20)&(entire['group']==grp)]['rmse'].mean()
        print(f"{name:<20} {grp:<10} {v80:>10.2f} {v20:>10.2f} {abs(v80-v20):>8.2f}")
