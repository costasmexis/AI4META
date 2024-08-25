from machinelearning.mlpipeline import MLPipelines
from machinelearning.mlexplain import MLExplainer
import pandas as pd

csv_dir_list = ['data/epic_lc_ms_pos.csv', 'data/gastric_cancer.csv','data/chronic_fatigue.csv', 'data/lung_cancer.csv', 'data/periodontal_inflammation.csv', 'data/epic_composite.csv']
evaluation_df = pd.DataFrame()
for csv_dir in csv_dir_list:
    if (csv_dir == 'data/epic_lc_ms_pos.csv'):
        label = 'group'
        estimators = ['ElasticNet','LogisticRegression','XGBClassifier','SVC','RandomForestClassifier','LinearDiscriminantAnalysis']
    elif (csv_dir == 'data/epic_composite.csv'):
        label = 'Factor1'
        estimators = ['ElasticNet','LogisticRegression','XGBClassifier','SVC','RandomForestClassifier','LinearDiscriminantAnalysis']
    else:
        label = 'Class'
        estimators=['ElasticNet','LogisticRegression','SVC','RandomForestClassifier']
    training_methods_list = ['one_sem','one_sem_grd','gso_1','gso_2']
    # training_methods_list = ['validation_score']
    
    mlpipe = MLPipelines(csv_dir=csv_dir, label=label)
    for estimator in estimators:
        for training_method in training_methods_list:
            mod, df = mlpipe.bayesian_search(estimator_name=estimator,scoring='matthews_corrcoef',boxplot=False, evaluation='bootstrap', n_trials=100, cv=5, warnings_filter=True,training_method=training_method, processors=15)
            df['Estimator'] = estimator
            df['Training Method'] = training_method
            df['Dataset'] = csv_dir
            evaluation_df = pd.concat([evaluation_df,df], ignore_index=True)
            print(f'FINISHED WITH {estimator} AND {training_method} AND {csv_dir}')

        evaluation_df.to_csv('paper_evaluation_rest.csv')