import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("figures/cap5", exist_ok=True)

files = {
    'REPLACE-BG':        'results/results_REPLACE_BG.parquet',
    'DiaTrend':          'results/results_DiaTrend.parquet',
    'T1DiabetesGranada': 'results/results_T1DiabetesGranada.parquet',
}

COLORS = {'male': '#1565C0', 'female': '#AD1457'}

# ── Fig 5.1: RMSE by glycaemic range ─────────────────────────────────────────
ranges_order = ['TBR_2', 'TBR_1', 'TIR', 'TAR_1', 'TAR_2']
range_labels = ['TBR-2\n(<54)', 'TBR-1\n(54-70)', 'TIR\n(70-180)',
                'TAR-1\n(181-250)', 'TAR-2\n(>250)']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('RMSE by glycaemic range across sex-proportion configurations',
             fontsize=14, fontweight='bold', y=1.01)

for ax, (name, fpath) in zip(axes, files.items()):
    df = pd.read_parquet(fpath)
    x_positions = np.arange(len(ranges_order))
    width = 0.35

    for grp in ['male', 'female']:
        offset = -width/2 if grp == 'male' else width/2
        bp_data = []
        for rng in ranges_order:
            # 11 values: one per configuration (averaged over 5 folds)
            vals = df[(df['Range'] == rng) & (df['group'] == grp)]\
                     .groupby('prop_M')['rmse'].mean().values
            bp_data.append(vals)

        ax.boxplot(bp_data,
                   positions=x_positions + offset,
                   widths=width * 0.85,
                   patch_artist=True,
                   showfliers=False,
                   medianprops=dict(color='white', linewidth=2),
                   whiskerprops=dict(color=COLORS[grp], linewidth=1.2),
                   capprops=dict(color=COLORS[grp], linewidth=1.2),
                   boxprops=dict(facecolor=COLORS[grp], alpha=0.7))

    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(range_labels, fontsize=9)
    ax.set_ylabel('RMSE (mg/dL)', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

male_patch   = mpatches.Patch(color=COLORS['male'],   alpha=0.7, label='Male test set')
female_patch = mpatches.Patch(color=COLORS['female'], alpha=0.7, label='Female test set')
fig.legend(handles=[male_patch, female_patch], loc='lower center', ncol=2,
           bbox_to_anchor=(0.5, -0.05), fontsize=10, frameon=False)

plt.tight_layout()
plt.savefig('figures/cap5/fig6_rmse_by_range.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: figures/cap5/fig6_rmse_by_range.png')


# ── Fig 5.2: Clarke Error Grid zone distribution ──────────────────────────────
zones       = ['zone_A', 'zone_B', 'zone_C', 'zone_D', 'zone_E']
zone_labels = ['Zone A', 'Zone B', 'Zone C', 'Zone D', 'Zone E']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Clarke Error Grid zone distribution across sex-proportion configurations',
             fontsize=14, fontweight='bold', y=1.01)

for ax, (name, fpath) in zip(axes, files.items()):
    df = pd.read_parquet(fpath)
    entire = df[df['Range'] == 'ENTIRE']
    x_positions = np.arange(len(zones))
    width = 0.35

    for grp in ['male', 'female']:
        offset = -width/2 if grp == 'male' else width/2
        bp_data = []
        for zone in zones:
            # 11 values: one per configuration (averaged over 5 folds)
            vals = entire[entire['group'] == grp]\
                        .groupby('prop_M')[zone].mean().values
            bp_data.append(vals)

        ax.boxplot(bp_data,
                   positions=x_positions + offset,
                   widths=width * 0.85,
                   patch_artist=True,
                   showfliers=False,
                   medianprops=dict(color='white', linewidth=2),
                   whiskerprops=dict(color=COLORS[grp], linewidth=1.2),
                   capprops=dict(color=COLORS[grp], linewidth=1.2),
                   boxprops=dict(facecolor=COLORS[grp], alpha=0.7))

    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(zone_labels, fontsize=9)
    ax.set_ylabel('Percentage (%)', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

male_patch   = mpatches.Patch(color=COLORS['male'],   alpha=0.7, label='Male test set')
female_patch = mpatches.Patch(color=COLORS['female'], alpha=0.7, label='Female test set')
fig.legend(handles=[male_patch, female_patch], loc='lower center', ncol=2,
           bbox_to_anchor=(0.5, -0.05), fontsize=10, frameon=False)

plt.tight_layout()
plt.savefig('figures/cap5/fig5_clarke_zones.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: figures/cap5/fig5_clarke_zones.png')

print('\n✓ Ambas figuras generadas.')