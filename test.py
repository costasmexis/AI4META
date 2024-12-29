from machinelearning.mlpipeline import MLPipelines
from machinelearning.mlexplain import MLExplainer
from dataloader.eda import DataExplorer
import pandas as pd

datasets = ['ICC']

for dataset in datasets:
    csv_dir = 'data/' + dataset + '.csv'
    if (dataset == 'epic_lc_ms_pos') or(dataset == 'epic_lc_ms_neg'):
        label = 'group'
    elif (dataset == 'epic_composite'):
        label = 'Factor1'
    elif (dataset == 'ICC'):
        label = 'type'
    elif (dataset == 'nhs_healthy') or (dataset == 'nhs_naive'):
        label = 'label'
    else:
        label = 'Class'
    
    print(f'STARTING WITH {dataset}')
    mlpipe = MLPipelines(label=label, csv_dir=csv_dir, database_name='ai4meta.db')
    mlpipe.nested_cv(parallel='freely_parallel', info_to_db=True)
    mlpipe.rcv_accel(info_to_db=True)
    print('Finished')

