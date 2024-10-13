from machinelearning.mlpipeline import MLPipelines
from machinelearning.mlexplain import MLExplainer
from dataloader.eda import DataExplorer
import pandas as pd

datasets = ['gastric_cancer', 'chronic_fatigue','lung_cancer','periodontal_inflammation','epic_lc_ms_pos','epic_composite']
inner_selection = ['one_sem','validation_score','gso_1','gso_2','one_sem_grd']
evaluation = ['cv_rounds','bootstrap','oob']
estimators = ['LogisticRegression','XGBClassifier','SVC','RandomForestClassifier','ElasticNet','LGBMClassifier','GradientBoostingClassifier','LinearDiscriminantAnalysis']

for dataset in datasets:
    dataset_df_final = pd.DataFrame()
    csv_dir = 'data/' + dataset + '.csv'
    if (dataset == 'epic_lc_ms_pos') or(dataset == 'epic_composite'):
        label = 'group'
    else:
        label= 'Class' 
    
    mlpipe = MLPipelines(label=label, csv_dir=csv_dir)
    for estimator in estimators:
        for evaluation_method in evaluation:
            for inner in inner_selection:
                mod, df = mlpipe.bayesian_search(estimator_name=estimator,scoring='matthews_corrcoef',boxplot=False, evaluation=evaluation_method, n_trials=100, cv=5, warnings_filter=True,training_method=inner,processors=4)
                dataset_df_final = pd.concat([dataset_df_final,df], ignore_index=True)
                dataset_df_final.to_csv('Final_Model_Results/' + dataset + '_final.csv')
                print(f'FINISHED WITH {dataset} AND {estimator} AND {evaluation_method} AND {inner}')
    
#     eda = DataExplorer(label=label, csv_dir=csv_dir)
#     feat = eda.statistical_difference(show_box=False)
#     mlpipe.X = mlpipe.X[feat]
#     for inner in inner_selection:
#         df = mlpipe.nested_cv(parallel='freely_parallel',inner_selection=inner,name_add='StatDiffTrial')
#         print(f'FINISHED WITH {dataset} AND {inner}')
#     print(f'FINISHED WITH {dataset} COMPLETLY')
# print('FINISHED ALL DATASETS')
