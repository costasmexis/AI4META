import pandas as pd
import numpy as np 
from machinelearning.mlpipeline import MLPipelines
from machinelearning.mlexplain import MLExplainer

csv_dir_list = ['data/periodontal_inflammation.csv']
training_methods_list = ['one_sem','one_sem_grd','gso_1','gso_2']
# training_methods_list = ['validation_score']
shap_values_list = ['M8','M18','M32']
estimators_list = [['ElasticNet','LogisticRegression','SVC','RandomForestClassifier'],['LinearDiscriminantAnalysis','LGBMClassifier','GradientBoostingClassifier']]

df_all = pd.read_csv('paper_evaluation_shap.csv', index_col=0)
for csv_dir in csv_dir_list:
    # if (csv_dir == 'data/gadric_cancer.csv'):
    label = 'Class'
    # if (csv_dir == 'data/gastric_cancer.csv'):
    #     shap_values = shap_values_list
    #     estimators2look = estimators_list[0]
    # else:
    #     shap_values = 'all'
    estimators2look = estimators_list[1]
    for training_method in training_methods_list:
        for estimator in estimators2look:
            mlpipe = MLPipelines(csv_dir=csv_dir, label=label)
            # if shap_values == 'all':
            #     shap_values = mlpipe.X.columns
            mod, df = mlpipe.bayesian_search(estimator_name=estimator,scoring='matthews_corrcoef',boxplot=False, evaluation='bootstrap', n_trials=100, cv=5, warnings_filter=True,training_method=training_method, processors=15)
            df['Estimator'] = estimator
            df['Training Method'] = training_method
            df['Dataset'] = csv_dir
            df['Features'] = 'all'
            df_all = pd.concat([df_all,df], ignore_index=True)
            print(f'FINISHED WITH {estimator} AND {training_method} AND {csv_dir} for SHAP features')
            
            df_all.to_csv('paper_evaluation_shap.csv')

df_all.to_csv('paper_evaluation_shap.csv')