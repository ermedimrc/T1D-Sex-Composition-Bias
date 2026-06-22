import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("figures", exist_ok=True)

# ── Rutas a los parquets ──────────────────────────────────────────────────────
files = {
    'REPLACE-BG':        'results/results_REPLACE_BG.parquet',
    'DiaTrend':          'results/results_DiaTrend.parquet',
    'T1DiabetesGranada': 'results/results_T1DiabetesGranada.parquet',
}

# ── Preparar datos agregados ──────────────────────────────────────────────────
dfs = {}
raw = {}
for name, f in files.items():
    df = pd.read_parquet(f)
    raw[name] = df
    entire = df[df['Range'] == 'ENTIRE']
    agg = entire.groupby(['prop_M', 'group'])[['mae', 'rmse', 'zone_AB']].agg(['mean', 'std']).reset_index()
    agg.columns = ['prop_M', 'group', 'mae_m', 'mae_s', 'rmse_m', 'rmse_s', 'zAB_m', 'zAB_s']
    dfs[name] = agg

props  = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
labels = ['100%H\n0%F', '90/10', '80/20', '70/30', '60/40', '50/50',
          '40/60', '30/70', '20/80', '10/90', '0%H\n100%F']

COLORS  = {'male': '#1565C0', 'female': '#AD1457', 'combined': '#2E7D32'}
MARKERS = {'male': 'o',       'female': 's',       'combined': 'D'}
LSTYLE  = {'male': '-',       'female': '-',       'combined': '--'}
LABELS  = {'male': 'Test: hombres', 'female': 'Test: mujeres', 'combined': 'Test: combinado'}


# ── Fig 1: MAE por proporción ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
fig.suptitle('MAE según proporción de sexo en entrenamiento',
             fontsize=14, fontweight='bold', y=1.01)

for ax, (name, agg) in zip(axes, dfs.items()):
    for grp in ['male', 'female', 'combined']:
        sub  = agg[agg['group'] == grp].sort_values('prop_M', ascending=False)
        vals = sub['mae_m'].values
        stds = sub['mae_s'].values
        ax.plot(range(len(props)), vals,
                color=COLORS[grp], linestyle=LSTYLE[grp],
                marker=MARKERS[grp], markersize=5, linewidth=1.8, zorder=3)
        ax.fill_between(range(len(props)), vals - stds, vals + stds,
                        color=COLORS[grp], alpha=0.08)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(props)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel('Proporción en training', fontsize=10)
    ax.set_ylabel('MAE (mg/dL)', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles = [mpatches.Patch(color=COLORS[g], label=LABELS[g]) for g in ['male', 'female', 'combined']]
fig.legend(handles=handles, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.06), fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('figures/fig1_mae_by_proportion.png', dpi=300, bbox_inches='tight')
plt.close()
print('Guardada: fig1_mae_by_proportion.png')


# ── Fig 2: Brecha H vs M ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Brecha de MAE entre grupos (mujeres − hombres)',
             fontsize=14, fontweight='bold', y=1.01)

for ax, (name, agg) in zip(axes, dfs.items()):
    male_mae   = agg[agg['group'] == 'male'].sort_values('prop_M', ascending=False)['mae_m'].values
    female_mae = agg[agg['group'] == 'female'].sort_values('prop_M', ascending=False)['mae_m'].values
    gap    = female_mae - male_mae
    colors = ['#AD1457' if g > 0 else '#1565C0' for g in gap]
    ax.bar(range(len(props)), gap, color=colors, width=0.6)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(np.mean(gap), color='gray', linewidth=1.2, linestyle='--',
               label=f'Media: {np.mean(gap):.2f} mg/dL')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(props)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Δ MAE (mg/dL)', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/fig2_mae_gap.png', dpi=300, bbox_inches='tight')
plt.close()
print('Guardada: fig2_mae_gap.png')


# ── Fig 3: MAE por rango glucémico ────────────────────────────────────────────
ranges_order = ['TBR_2', 'TBR_1', 'TIR', 'TAR_1', 'TAR_2', 'ENTIRE']
range_labels = ['TBR-2\n(<54)', 'TBR-1\n(54-70)', 'TIR\n(70-180)',
                'TAR-1\n(181-250)', 'TAR-2\n(>250)', 'ENTIRE']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('MAE por rango glucémico (media sobre todas las proporciones)',
             fontsize=14, fontweight='bold', y=1.01)

x     = np.arange(len(ranges_order))
width = 0.35

for ax, (name, fpath) in zip(axes, files.items()):
    df        = pd.read_parquet(fpath)
    range_agg = df.groupby(['Range', 'group'])['mae'].mean().reset_index()

    def get_val(r, grp):
        row = range_agg[(range_agg['Range'] == r) & (range_agg['group'] == grp)]
        return row['mae'].values[0] if len(row) > 0 else 0

    male_vals   = [get_val(r, 'male')   for r in ranges_order]
    female_vals = [get_val(r, 'female') for r in ranges_order]

    ax.bar(x - width / 2, male_vals,   width, label='Hombres', color='#1565C0', alpha=0.85)
    ax.bar(x + width / 2, female_vals, width, label='Mujeres',  color='#AD1457', alpha=0.85)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels, fontsize=8)
    ax.set_ylabel('MAE (mg/dL)', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/fig3_mae_by_range.png', dpi=300, bbox_inches='tight')
plt.close()
print('Guardada: fig3_mae_by_range.png')


# ── Fig 4: Zona A+B por proporción ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
fig.suptitle('Zona A+B (%) según proporción de sexo en entrenamiento',
             fontsize=14, fontweight='bold', y=1.01)

for ax, (name, agg) in zip(axes, dfs.items()):
    for grp in ['male', 'female', 'combined']:
        sub  = agg[agg['group'] == grp].sort_values('prop_M', ascending=False)
        vals = sub['zAB_m'].values
        stds = sub['zAB_s'].values
        ax.plot(range(len(props)), vals,
                color=COLORS[grp], linestyle=LSTYLE[grp],
                marker=MARKERS[grp], markersize=5, linewidth=1.8, zorder=3)
        ax.fill_between(range(len(props)), vals - stds, vals + stds,
                        color=COLORS[grp], alpha=0.08)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(props)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel('Proporción en training', fontsize=10)
    ax.set_ylabel('Zona A+B (%)', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles = [mpatches.Patch(color=COLORS[g], label=LABELS[g]) for g in ['male', 'female', 'combined']]
fig.legend(handles=handles, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.06), fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('figures/fig4_zoneAB_by_proportion.png', dpi=300, bbox_inches='tight')
plt.close()
print('Guardada: fig4_zoneAB_by_proportion.png')


# ── Fig 5: Comparativa 3 datasets ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
datasets = list(dfs.keys())
x     = np.arange(len(datasets))
width = 0.25

for i, grp in enumerate(['male', 'female', 'combined']):
    means = [dfs[ds][dfs[ds]['group'] == grp]['mae_m'].mean() for ds in datasets]
    stds  = [dfs[ds][dfs[ds]['group'] == grp]['mae_m'].std()  for ds in datasets]
    ax.bar(x + i * width, means, width, label=LABELS[grp],
           color=COLORS[grp], alpha=0.85, yerr=stds, capsize=4)

ax.set_title('MAE medio por dataset y grupo (media sobre todas las proporciones)',
             fontsize=12, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylabel('MAE (mg/dL)', fontsize=10)
ax.legend(fontsize=9, frameon=False)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('figures/fig5_dataset_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print('Guardada: fig5_dataset_comparison.png')


# ── Fig 6: MAE por fold ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('MAE por fold — variabilidad del 5-fold cross-validation',
             fontsize=14, fontweight='bold', y=1.01)

for ax, (name, df) in zip(axes, raw.items()):
    entire   = df[df['Range'] == 'ENTIRE']
    fold_agg = entire.groupby(['fold', 'group'])['mae'].mean().reset_index()
    for grp in ['male', 'female']:
        sub = fold_agg[fold_agg['group'] == grp].sort_values('fold')
        ax.plot(sub['fold'].values, sub['mae'].values,
                color=COLORS[grp], marker=MARKERS[grp],
                markersize=7, linewidth=1.8, label=LABELS[grp])
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(['Fold 0', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4'], fontsize=9)
    ax.set_ylabel('MAE (mg/dL)', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/fig6_mae_by_fold.png', dpi=300, bbox_inches='tight')
plt.close()
print('Guardada: fig6_mae_by_fold.png')


# ── Fig 7: RMSE por proporción ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
fig.suptitle('RMSE según proporción de sexo en entrenamiento',
             fontsize=14, fontweight='bold', y=1.01)

for ax, (name, agg) in zip(axes, dfs.items()):
    for grp in ['male', 'female', 'combined']:
        sub  = agg[agg['group'] == grp].sort_values('prop_M', ascending=False)
        vals = sub['rmse_m'].values
        stds = sub['rmse_s'].values
        ax.plot(range(len(props)), vals,
                color=COLORS[grp], linestyle=LSTYLE[grp],
                marker=MARKERS[grp], markersize=5, linewidth=1.8, zorder=3)
        ax.fill_between(range(len(props)), vals - stds, vals + stds,
                        color=COLORS[grp], alpha=0.08)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(props)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel('Proporción en training', fontsize=10)
    ax.set_ylabel('RMSE (mg/dL)', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles = [mpatches.Patch(color=COLORS[g], label=LABELS[g]) for g in ['male', 'female', 'combined']]
fig.legend(handles=handles, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.06), fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('figures/fig7_rmse_by_proportion.png', dpi=300, bbox_inches='tight')
plt.close()
print('Guardada: fig7_rmse_by_proportion.png')

print('\n✓ Todas las figuras generadas.')

# ── Fig 8: Brecha H vs M en Zone A+B ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Sex gap in zone A+B (female minus male) by sex-proportion configuration',
             fontsize=14, fontweight='bold', y=1.01)
for ax, (name, agg) in zip(axes, dfs.items()):
    male_ab   = agg[agg['group'] == 'male'].sort_values('prop_M', ascending=False)['zAB_m'].values
    female_ab = agg[agg['group'] == 'female'].sort_values('prop_M', ascending=False)['zAB_m'].values
    gap    = female_ab - male_ab
    mean_gap = np.mean(gap)
    std_gap  = np.std(gap)
    colors = ['#1565C0' if g > 0 else '#AD1457' for g in gap]
    ax.bar(range(len(props)), gap, color=colors, width=0.6)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(mean_gap, color='gray', linewidth=1.2, linestyle='--')
    ax.fill_between([-0.5, len(props)-0.5],
                    mean_gap - std_gap,
                    mean_gap + std_gap,
                    alpha=0.15, color='gray')
    ax.text(0.98, 0.02, f'Mean = {mean_gap:.2f}  SD = {std_gap:.2f} pp',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='gray')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(props)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Δ Zone A+B (pp)', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
pink_patch = mpatches.Patch(color='#AD1457', label='Female harder to predict')
blue_patch = mpatches.Patch(color='#1565C0', label='Male harder to predict')
mean_line  = plt.Line2D([0], [0], color='gray', linestyle='--', label='Mean gap')
sd_patch   = mpatches.Patch(color='gray', alpha=0.3, label='±1 SD')
fig.legend(handles=[pink_patch, blue_patch, mean_line, sd_patch], loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.06), fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('figures/fig8_zoneAB_gap.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: figures/fig8_zoneAB_gap.png')

# ── Fig 9: Brecha H vs M en RMSE ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Sex gap in RMSE (female minus male) by sex-proportion configuration',
             fontsize=14, fontweight='bold', y=1.01)
for ax, (name, df) in zip(axes, raw.items()):
    entire = df[df['Range'] == 'ENTIRE']
    gaps = []
    for prop in props:
        m = entire[(entire['prop_M'] == prop) & (entire['group'] == 'male')]['rmse'].mean()
        f = entire[(entire['prop_M'] == prop) & (entire['group'] == 'female')]['rmse'].mean()
        gaps.append(f - m)
    gaps = np.array(gaps)
    mean_gap = np.mean(gaps)
    std_gap  = np.std(gaps)
    colors = ['#AD1457' if g > 0 else '#1565C0' for g in gaps]
    ax.bar(range(len(props)), gaps, color=colors, width=0.6)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(mean_gap, color='gray', linewidth=1.2, linestyle='--')
    ax.fill_between([-0.5, len(props)-0.5],
                    mean_gap - std_gap,
                    mean_gap + std_gap,
                    alpha=0.15, color='gray')
    ax.text(0.98, 0.02, f'Mean = {mean_gap:.2f}  SD = {std_gap:.2f} mg/dL',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='gray')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(props)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Δ RMSE (mg/dL)', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
pink_patch = mpatches.Patch(color='#AD1457', label='Female harder to predict')
blue_patch = mpatches.Patch(color='#1565C0', label='Male harder to predict')
mean_line  = plt.Line2D([0], [0], color='gray', linestyle='--', label='Mean gap')
sd_patch   = mpatches.Patch(color='gray', alpha=0.3, label='±1 SD')
fig.legend(handles=[pink_patch, blue_patch, mean_line, sd_patch], loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.06), fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('figures/cap5/fig7_rmse_gap.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: figures/cap5/fig7_rmse_gap.png')