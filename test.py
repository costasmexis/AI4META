from machinelearning.mlpipeline import MLPipelines
from machinelearning.mlexplain import MLExplainer
from dataloader.eda import DataExplorer
import pandas as pd

datasets = ['epic_lc_ms_pos', 'epic_composite']#, 'chronic_fatigue','lung_cancer','periodontal_inflammation','epic_lc_ms_pos','epic_composite']

for dataset in datasets:
    csv_dir = 'data/' + dataset + '.csv'
    if dataset == 'epic_lc_ms_pos':
        label = 'group'
    else:
        label= 'Factor_1' 
    
    print(f'STARTING WITH {dataset}')
    mlpipe = MLPipelines(label=label, csv_dir=csv_dir)
    mlpipe.nested_cv(parallel='freely_parallel', info_to_db=True)
    # mlpipe = MLPipelines(label=label, csv_dir=csv_dir)
    # for estimator in estimators:
    #     for evaluation_method in evaluation:
    #         for inner in inner_selection:
    #             mod, df = mlpipe.bayesian_search(estimator_name=estimator,scoring='matthews_corrcoef',boxplot=False, evaluation=evaluation_method, n_trials=100, cv=5, warnings_filter=True,training_method=inner,processors=4)
    #             dataset_df_final = pd.concat([dataset_df_final,df], ignore_index=True)
    #             dataset_df_final.to_csv('Final_Model_Results/' + dataset + '_final.csv')
    #             print(f'FINISHED WITH {dataset} AND {estimator} AND {evaluation_method} AND {inner}')
    
#     eda = DataExplorer(label=label, csv_dir=csv_dir)
#     feat = eda.statistical_difference(show_box=False)
#     mlpipe.X = mlpipe.X[feat]
#     for inner in inner_selection:
#         df = mlpipe.nested_cv(parallel='freely_parallel',inner_selection=inner,name_add='StatDiffTrial')
#         print(f'FINISHED WITH {dataset} AND {inner}')
#     print(f'FINISHED WITH {dataset} COMPLETLY')
# print('FINISHED ALL DATASETS')
