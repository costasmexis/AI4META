import pandas as pd
import numpy as np 
from machinelearning.mlpipeline import MLPipelines
from machinelearning.mlexplain import MLExplainer
from tqdm import tqdm

csv_dir_list = ['data/chronic_fatigue.csv','data/lung_cancer.csv']

trme_per_dataset = [['validation_score','one_sem'],['validation_score','one_sem','one_sem_grd','gso_1']]
estimators_per_dataset = [['LogisticRegression','SVC','LinearDiscriminantAnalysis'],['LogisticRegression','ElasticNet','SVC']]

shap_per_dataset = [[['M5','M25','M15','M23','M8','M12','M19','M17'],['M5','M25','M15','M23','M8','M12'],['M5','M25','M23','M12','M26']],[['M121','M116','M7','M57','M64']]]

label = 'Class'

# df_all = pd.read_csv('papers_replication.csv')

df_all = pd.DataFrame()

for csv_dir in csv_dir_list:
    if (csv_dir == 'data/chronic_fatigue.csv'):
        shap_values_list = shap_per_dataset[0]
        training_method_list = trme_per_dataset[0]
        estimators_list = estimators_per_dataset[0]
    else:
        shap_values_list = shap_per_dataset[1]
        training_method_list = trme_per_dataset[1]
        estimators_list = estimators_per_dataset[1]
        
    mlpipe = MLPipelines(csv_dir=csv_dir, label=label)
    for training_method in training_method_list:
        if (csv_dir == 'data/chronic_fatigue.csv'):
            mod, df = mlpipe.bayesian_search(estimator_name='GradientBoostingClassifier',scoring='matthews_corrcoef',boxplot=False, evaluation='bootstrap', n_trials=100, cv=5, warnings_filter=True,training_method=training_method, processors=15)
            df['Features'] = 'all'
            df['Estimator'] = 'GradientBoostingClassifier'
            df['Training Method'] = training_method
            df['Dataset'] = csv_dir

            df_all = pd.concat([df_all,df], ignore_index=True)

        for estimator in estimators_list:
            for shap_value in shap_values_list:
                print(f'LOADING {csv_dir} WITH {training_method} AND {estimator} AND {shap_value}')
                mod, df = mlpipe.bayesian_search(features_names_list=shap_value,estimator_name=estimator,scoring='matthews_corrcoef',boxplot=False, evaluation='bootstrap', n_trials=100, cv=5, warnings_filter=True,training_method=training_method, processors=15)
                df['Features'] = str(shap_value)
                df['Estimator'] = estimator
                df['Training Method'] = training_method
                df['Dataset'] = csv_dir
                df_all = pd.concat([df_all,df], ignore_index=True)
                
                df_all.to_csv('papers_replication_extra.csv')
        
    print(f'COMPLETED {csv_dir}!!!')

    df_all.to_csv('papers_replication_extra.csv')