import pandas as pd
import numpy as np
from scipy import stats

files = {
    'REPLACE-BG': '../results/results_REPLACE_BG.parquet',
    'DiaTrend': '../results/results_DiaTrend.parquet',
    'T1DiabetesGranada': '../results/results_T1DiabetesGranada.parquet',
}

props = [0,10,20,30,40,50,60,70,80,90,100]

for name, fpath in files.items():
    df = pd.read_parquet(fpath)
    entire = df[df['Range']=='ENTIRE']
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")

    for grp in ['male','female']:
        print(f"\n  Group: {grp}")

        # Friedman: configurations as treatments, folds as blocks
        # Matrix: rows=folds, cols=configurations
        for metric in ['rmse','zone_AB']:
            data_matrix = []
            for prop in props:
                fold_vals = entire[(entire['prop_M']==prop)&(entire['group']==grp)].sort_values('fold')[metric].values
                data_matrix.append(fold_vals)
            data_matrix = np.array(data_matrix).T  # shape: (folds, configs)
            stat, p = stats.friedmanchisquare(*data_matrix.T)
            print(f"    Friedman {metric}: chi2={stat:.2f}, p={p:.3f}")

        # Spearman: correlation between prop_M and mean performance
        for metric in ['rmse','zone_AB']:
            means = [entire[(entire['prop_M']==p)&(entire['group']==grp)][metric].mean() for p in props]
            rho, p = stats.spearmanr(props, means)
            print(f"    Spearman {metric}: rho={rho:.3f}, p={p:.3f}")

    # Wilcoxon RQ3: degradation male (50->0F) vs degradation female (50->100M)
    print(f"\n  Wilcoxon RQ3:")
    for metric in ['rmse','zone_AB']:
        male_50  = entire[(entire['prop_M']==50)&(entire['group']=='male')].sort_values('fold')[metric].values
        male_0   = entire[(entire['prop_M']==0) &(entire['group']=='male')].sort_values('fold')[metric].values
        female_50= entire[(entire['prop_M']==50)&(entire['group']=='female')].sort_values('fold')[metric].values
        female_100=entire[(entire['prop_M']==100)&(entire['group']=='female')].sort_values('fold')[metric].values
        delta_m = male_0   - male_50    # degradation male
        delta_f = female_100 - female_50  # degradation female
        stat, p = stats.wilcoxon(delta_m, delta_f)
        print(f"    {metric}: delta_M={delta_m.mean():.3f}±{delta_m.std():.3f}, delta_F={delta_f.mean():.3f}±{delta_f.std():.3f}, p={p:.3f}")
