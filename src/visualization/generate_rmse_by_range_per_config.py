import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures/cap5", exist_ok=True)
os.makedirs("figures/appendix_rmse_by_range", exist_ok=True)

files = {
    'REPLACE-BG':        'results/results_REPLACE_BG.parquet',
    'DiaTrend':          'results/results_DiaTrend.parquet',
    'T1DiabetesGranada': 'results/results_T1DiabetesGranada.parquet',
}

props = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
configs = [f'{p}M_{100-p}F' for p in props]

ranges_order = ['TBR_2', 'TBR_1', 'TIR', 'TAR_1', 'TAR_2']
range_labels = ['TBR-2\n(<54)', 'TBR-1\n(54-70)', 'TIR\n(70-180)',
                'TAR-1\n(181-250)', 'TAR-2\n(>250)']

# Cargar todos los datasets una vez
dfs = {name: pd.read_parquet(path) for name, path in files.items()}

x = np.arange(len(ranges_order))
width = 0.35

COLOR_MALE = '#1565C0'
COLOR_FEMALE = '#AD1457'


# ─────────────────────────────────────────────────────────────────
# FIGURA 5.1 (cuerpo principal): promedio sobre todas las configuraciones
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('RMSE by glycaemic range (averaged over all sex-proportion configurations)',
             fontsize=14, fontweight='bold', y=1.02)

for ax, (name, df) in zip(axes, dfs.items()):
    # Promediar sobre TODAS las configuraciones y folds (no filtramos por label)
    range_agg = df.groupby(['Range', 'group'])['rmse'].mean().reset_index()

    def get_val(r, grp):
        row = range_agg[(range_agg['Range'] == r) & (range_agg['group'] == grp)]
        return row['rmse'].values[0] if len(row) > 0 else 0

    male_vals = [get_val(r, 'male') for r in ranges_order]
    female_vals = [get_val(r, 'female') for r in ranges_order]

    ax.bar(x - width/2, male_vals, width, label='Male', color=COLOR_MALE, alpha=0.85)
    ax.bar(x + width/2, female_vals, width, label='Female', color=COLOR_FEMALE, alpha=0.85)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels, fontsize=8)
    ax.set_ylabel('RMSE (mg/dL)', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/cap5/fig6_rmse_by_range.png', dpi=300, bbox_inches='tight')
plt.close()
print('Guardada: figures/cap5/fig6_rmse_by_range.png')


# ─────────────────────────────────────────────────────────────────
# APÉNDICE B: una figura por cada una de las 11 configuraciones,
# con eje Y estandarizado para permitir comparación visual directa
# ─────────────────────────────────────────────────────────────────

# Calcular el máximo global de RMSE para fijar el eje Y en todas las figuras
global_max = 0
for name, df in dfs.items():
    for config in configs:
        df_config = df[df['label'] == config]
        range_agg = df_config.groupby(['Range', 'group'])['rmse'].mean()
        if len(range_agg) > 0:
            global_max = max(global_max, range_agg.max())

y_limit = global_max * 1.1  # 10% de margen arriba
print(f'Eje Y fijado a: 0 - {y_limit:.1f} mg/dL')

for config in configs:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'RMSE by glycaemic range — Configuration: {config.replace("_", "/")}',
                 fontsize=14, fontweight='bold', y=1.02)

    for ax, (name, df) in zip(axes, dfs.items()):
        df_config = df[df['label'] == config]
        range_agg = df_config.groupby(['Range', 'group'])['rmse'].mean().reset_index()

        def get_val(r, grp):
            row = range_agg[(range_agg['Range'] == r) & (range_agg['group'] == grp)]
            return row['rmse'].values[0] if len(row) > 0 else 0

        male_vals = [get_val(r, 'male') for r in ranges_order]
        female_vals = [get_val(r, 'female') for r in ranges_order]

        ax.bar(x - width/2, male_vals, width, label='Male', color=COLOR_MALE, alpha=0.85)
        ax.bar(x + width/2, female_vals, width, label='Female', color=COLOR_FEMALE, alpha=0.85)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(range_labels, fontsize=8)
        ax.set_ylabel('RMSE (mg/dL)', fontsize=10)
        ax.set_ylim(0, y_limit)  # eje Y fijo en todas las figuras
        ax.legend(fontsize=9, frameon=False)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = f'figures/appendix_rmse_by_range/fig_rmse_range_{config}.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Guardada: {out_path}')

print('\n✓ Figura 5.1 y 11 figuras del apéndice generadas con eje Y estandarizado.')