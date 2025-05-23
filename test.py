from src.models.mlselection import MLPipelines
# from machinelearning.mlexplain import MLExplainer
# from dataloader.eda import DataExplorer
# import pandas as pd
datasets = ['ICC']

for dataset in datasets:
    csv_dir = 'data/processed/' + dataset + '.csv'
    # if dataset == 'epic_lc_ms_pos':
    #     label = 'group'
    # elif dataset == 'epic_composite':
    #     label= 'Factor1' 
    # else:
    label='type'
    # label = 'Class'
    
    print(f'STARTING WITH {dataset}')
    mlpipe = MLPipelines(label=label, csv_dir=csv_dir)
    mlpipe.nested_cv(num_features=20)
    mlpipe.rcv_accel(num_features=20)  
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
