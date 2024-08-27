import pandas as pd
import numpy as np 
from machinelearning.mlpipeline import MLPipelines
from machinelearning.mlexplain import MLExplainer

csv_dir_list = ['data/epic_lc_ms_pos.csv', 'data/gastric_cancer.csv']
training_methods_list = ['one_sem','one_sem_grd','gso_1','gso_2']
# training_methods_list = ['validation_score']
shap_values_list = [['M8','M18','M32'],['942.9824_0.57', '467.3822_23.01', '393.3454_25.66', '613.4767_25.19', '379.3289_23.95']]
estimators_list = ['GradientBoostingClassifier','LGBMClassifier']

df_all = pd.DataFrame()

for csv_dir in csv_dir_list:
    if (csv_dir == 'data/epic_lc_ms_pos.csv'):
        label = 'group'
        shap_values = shap_values_list[1]
    else:
        label= 'Class' 
        shap_values = shap_values_list[0]
    for training_method in training_methods_list:
        for estimator in estimators_list:
            mlpipe = MLPipelines(csv_dir=csv_dir, label=label)
            mod, df = mlpipe.bayesian_search(estimator_name=estimator,scoring='matthews_corrcoef',boxplot=False, evaluation='bootstrap', n_trials=100, cv=5, warnings_filter=True,training_method=training_method, processors=15)
            df['Estimator'] = estimator
            df['Training Method'] = training_method
            df['Dataset'] = csv_dir
            df['Features'] = 'all'
            df_all = pd.concat([df_all,df], ignore_index=True)
            print(f'FINISHED WITH {estimator} AND {training_method} AND {csv_dir} for ALL features')
            mlpipe.X = mlpipe.X[shap_values]
            mod, df = mlpipe.bayesian_search(estimator_name=estimator,scoring='matthews_corrcoef',boxplot=False, evaluation='bootstrap', n_trials=100, cv=5, warnings_filter=True,training_method=training_method, processors=15)
            df['Estimator'] = estimator
            df['Training Method'] = training_method
            df['Dataset'] = csv_dir
            df['Features'] = str(shap_values)
            df_all = pd.concat([df_all,df], ignore_index=True)
            print(f'FINISHED WITH {estimator} AND {training_method} AND {csv_dir} for SHAP features')
            
            df_all.to_csv('paper_evaluation_shap.csv')

df_all.to_csv('paper_evaluation_shap.csv')