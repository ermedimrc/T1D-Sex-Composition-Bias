import pandas as pd

datasets = {
    'REPLACE-BG': 'results/results_REPLACE_BG.parquet',
    'DiaTrend': 'results/results_DiaTrend.parquet',
    'T1DiabetesGranada': 'results/results_T1DiabetesGranada.parquet',
}

props = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
configs = [f'{p}M_{100-p}F' for p in props]
ranges_order = ['TBR_2', 'TBR_1', 'TIR', 'TAR_1', 'TAR_2']
range_labels = {'TBR_2': 'TBR-2 (<54 mg/dL)', 'TBR_1': 'TBR-1 (54-70 mg/dL)', 
                'TIR': 'TIR (70-180 mg/dL)', 'TAR_1': 'TAR-1 (181-250 mg/dL)', 
                'TAR_2': 'TAR-2 (>250 mg/dL)'}

output_lines = []

for dataset_name, path in datasets.items():
    df = pd.read_parquet(path)
    
    for r in ranges_order:
        df_range = df[df['Range'] == r]
        agg = df_range.groupby(['label', 'group'])['zone_AB'].agg(['mean', 'std']).reset_index()
        
        lines = []
        lines.append(r'\begin{table}[ht]')
        lines.append(r'\centering')
        lines.append(rf'\caption{{Clarke Error Grid zone A+B on {dataset_name}, {range_labels[r]} (\%, mean~$\pm$~std across 5 folds).}}')
        lines.append(rf'\label{{tab:{dataset_name.lower().replace("-","")}_{r.lower()}}}')
        lines.append(r'\begin{tabular}{l ccc}')
        lines.append(r'\hline')
        lines.append(r'\textbf{Config.} & \textbf{Male} & \textbf{Female} & \textbf{Combined} \\')
        lines.append(r'\hline')
        
        for config in configs:
            row_data = agg[agg['label'] == config]
            
            def get_val(grp):
                sub = row_data[row_data['group'] == grp]
                if len(sub) == 0:
                    return 'N/A'
                m, s = sub['mean'].values[0], sub['std'].values[0]
                return f'${m:.3f} \\pm {s:.3f}$'
            
            male_val = get_val('male')
            female_val = get_val('female')
            combined_val = get_val('combined')
            
            config_label = config.replace('_', r'\_')
            lines.append(f'{config_label} & {male_val} & {female_val} & {combined_val} \\\\')
        
        lines.append(r'\hline')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')
        lines.append('')
        
        output_lines.extend(lines)

with open('tables_by_range.tex', 'w') as f:
    f.write('\n'.join(output_lines))

print(f'Generadas {len(datasets) * len(ranges_order)} tablas en tables_by_range.tex')