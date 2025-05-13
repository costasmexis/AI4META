from src.models.mlselection import MLPipelines
datasets = ['epic_composite']

for dataset in datasets:
    csv_dir = 'data/processed/' + dataset + '.csv'
    if dataset == 'epic_lc_ms_pos':
        label = 'group'
    elif dataset == 'epic_composite':
        label= 'Factor1' 
    mlpipe = MLPipelines(label=label, csv_dir=csv_dir)
    for num_feature in [None]:
        for estimator in ['CatBoostClassifier','XGBClassifier','ElasticNet']:
            for inner in ['validation_score','one_sem','one_sem_grd','gso_1','gso_2']:
                for evaluation in ['bootstrap', 'oob', 'cv_rounds']:
                    mlpipe.search_cv(estimator_name=estimator,num_features=num_feature,inner_selection=inner,evaluation=evaluation, processors=5)

# mlpipe.search_cv(estimator_name=estimator,num_features=num_feature,inner_selection=inner,evaluation=evaluation)

# datasets = ['ICC']

# # num_features = [10,20]#30,40,50,None]
# # estimators = ['LGBMClassifier','XGBClassifier','RandomForestClassifier','ElasticNet','GradientBoostingClassifier']
# # inner_methods = ['validation_score','one_sem','one_sem_grd','gso_1','gso_2']
# # evaluation_methods = ['bootstrap', 'oob', 'cv_rounds']
# # datasets = ['chronic_fatigue','lung_cancer','periodontal_inflammation','gastric_cancer']#,'epic_composite']

# for dataset in datasets:
#     csv_dir = 'data/processed/' + dataset + '.csv'
#     if dataset == 'epic_lc_ms_pos':
#         label = 'group'
#         mlpipe = MLPipelines(label=label, csv_dir=csv_dir, database_name="epic_lcmspos.db")
#     elif dataset == 'epic_composite':
#         label= 'Factor1' 
#         mlpipe = MLPipelines(label=label, csv_dir=csv_dir, database_name="epic_composite.db")
#     elif dataset == 'ICC':
#         label='type'
#         mlpipe = MLPipelines(label=label, csv_dir=csv_dir, database_name="icc.db")
#     else:
#         label = 'Class'
#         if dataset == 'chronic_fatigue':
#             mlpipe = MLPipelines(label=label, csv_dir=csv_dir, database_name="chronic_fatigue.db")
#         elif dataset == 'lung_cancer':
#             mlpipe = MLPipelines(label=label, csv_dir=csv_dir, database_name="lung_cancer.db")
#         elif dataset == 'periodontal_inflammation':
#             mlpipe = MLPipelines(label=label, csv_dir=csv_dir, database_name="periodontal_inflammation.db")
#         elif dataset == 'gastric_cancer':
#             mlpipe = MLPipelines(label=label, csv_dir=csv_dir, database_name="gastric_cancer.db")
    
#     # for num_feature in num_features:
#     #     for estimator in estimators:
#     #         for inner in inner_methods:
#     #             for evaluation in evaluation_methods:
#                     # mlpipe.search_cv(estimator_name=estimator,num_features=num_feature,inner_selection=inner,evaluation=evaluation)
#     # mlpipe.rcv_accel(num_features=[20,30,40,50,100],info_to_db=True)
#     mlpipe.nested_cv(num_features=[50,100],info_to_db=True,)
#     print(f'FINISHED WITH {dataset}')
