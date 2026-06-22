import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Rutas a los parquets ──────────────────────────────────────────────────────
files = {
    'REPLACE-BG':        'results/results_REPLACE_BG.parquet',
    'DiaTrend':          'results/results_DiaTrend.parquet',
    'T1DiabetesGranada': 'results/results_T1DiabetesGranada.parquet',
}

# ── Preparar datos agregados ──────────────────────────────────────────────────
dfs = {}
for name, f in files.items():
    df = pd.read_parquet(f)
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

# ── Fig 1: MAE por proporción (3 datasets) ───────────────────────────────────
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
                marker=MARKERS[grp], markersize=5, linewidth=1.8,
                label=LABELS[grp], zorder=3)
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
plt.savefig('fig1_mae_by_proportion.png', dpi=300, bbox_inches='tight')
plt.close()
print('Guardada: fig1_mae_by_proportion.png')

# ── Fig 2: Brecha H vs M por dataset ─────────────────────────────────────────
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
plt.savefig('fig2_mae_gap.png', dpi=300, bbox_inches='tight')
plt.close()
print('Guardada: fig2_mae_gap.png')

# ── Fig 3: MAE por rango glucémico ────────────────────────────────────────────
ranges_order  = ['TBR_2',    'TBR_1',    'TIR',       'TAR_1',      'TAR_2',    'ENTIRE']
range_labels  = ['TBR-2\n(<54)', 'TBR-1\n(54-70)', 'TIR\n(70-180)',
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
plt.savefig('fig3_mae_by_range.png', dpi=300, bbox_inches='tight')
plt.close()
print('Guardada: fig3_mae_by_range.png')

print('\n✓ Todas las figuras generadas.')
