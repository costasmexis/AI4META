from machinelearning.mlpipeline import MLPipelines
from machinelearning.mlexplain import MLExplainer
from dataloader.eda import DataExplorer

datasets = ['gastric_cancer', 'chronic_fatigue','lung_cancer','periodontal_inflammation','epic_lc_ms_neg','epic_lc_ms_pos','epic_ce_ms']
inner_selection = ['one_sem','validation_score','gso_1','gso_2','one_sem_grd']

for dataset in datasets:
    csv_dir = 'data/' + dataset + '.csv'
    if (dataset == 'epic_ce_ms') or (dataset == 'epic_lc_ms_neg') or (dataset == 'epic_lc_ms_pos'):
        label = 'group'
    else:
        label= 'Class' 
    
    mlpipe = MLPipelines(label=label, csv_dir=csv_dir)
    eda = DataExplorer(label=label, csv_dir=csv_dir)
    feat = eda.statistical_difference(show_box=False)
    mlpipe.X = mlpipe.X[feat]
    for inner in inner_selection:
        df = mlpipe.nested_cv(parallel='freely_parallel',inner_selection=inner,name_add='StatDiffTrial')
        print(f'FINISHED WITH {dataset} AND {inner}')
    print(f'FINISHED WITH {dataset} COMPLETLY')
print('FINISHED ALL DATASETS')
