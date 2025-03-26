from src.models.mlselection import MLPipelines

datasets = ['epic_lc_ms_pos']#,'epic_composite']
num_features = [10]
estimators = ['ElasticNet']#,'XGBClassifier','CatBoostClassifier']
inner_methods = ['validation_score']#,'one_sem','one_sem_grd','gso_1','gso_2']
evaluation_methods = ['bootstrap', 'oob', 'cv_rounds', 'train_test']



for dataset in datasets:
    csv_dir = 'data/processed/' + dataset + '.csv'
    if dataset == 'epic_lc_ms_pos':
        label = 'group'
    elif dataset == 'epic_composite':
        label= 'Factor1' 
    else:
        label='type'
    mlpipe = MLPipelines(label=label, csv_dir=csv_dir)

    for num_feature in num_features:
        for estimator in estimators:
            for inner in inner_methods:
                for evaluation in evaluation_methods:
                    # print(f'STARTING WITH {dataset} AND {estimator} AND {num_feature} AND {inner} AND {evaluation}')
                    # mlpipe.search_cv(estimator_name=estimator,num_features=num_feature,inner_selection=inner,evaluation=evaluation)
                    # mlpipe.rcv_accel(rounds=2,search_on=[estimator],num_features=num_feature,class_balance='smote', info_to_db=True)
                    mlpipe.nested_cv(search_on=[estimator],num_features=num_feature,info_to_db=True)