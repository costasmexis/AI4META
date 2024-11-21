import pandas as pd
import numpy as np 
from machinelearning.mlpipeline import MLPipelines
from machinelearning.mlexplain import MLExplainer
from tqdm import tqdm

# csv_dir_list = ['data/chronic_fatigue.csv','data/lung_cancer.csv']
csv_dir = 'data/epic_lc_ms_pos.csv'
training_methods_list = ['one_sem']
estimators_list = ['ElasticNet','SVC']

shap_values_list = [['942.9824_0.57', '467.3822_23.01', '393.3454_25.66', '613.4767_25.19', '379.3289_23.95','274.2498_23.94','569.4492_25.31'],['942.9824_0.57', '393.3454_25.66', '613.4767_25.19', '379.3289_23.95']]

label = 'group'

mlpipe = MLPipelines(csv_dir=csv_dir, label=label)
df_all = pd.read_csv('papers_replication.csv')

for training_method in training_methods_list:
    for estimator in estimators_list:
        for shap_value in shap_values_list:
            print(f'LOADING {csv_dir} WITH {training_method} AND {estimator} AND {shap_value}')
            mod, df = mlpipe.bayesian_search(features_names_list=shap_value,estimator_name=estimator,scoring='matthews_corrcoef',boxplot=False, evaluation='bootstrap', n_trials=100, cv=5, warnings_filter=True,training_method=training_method, processors=10)
            df['Features'] = str(shap_value)
            df['Estimator'] = estimator
            df['Training Method'] = training_method
            df['Dataset'] = csv_dir
            df_all = pd.concat([df_all,df], ignore_index=True)
            
            df_all.to_csv('papers_replication.csv')

df_all.to_csv('papers_replication.csv')