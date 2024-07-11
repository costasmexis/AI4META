from machinelearning.mlpipeline import MLPipelines
from machinelearning.mlexplain import MLExplainer

csv_dir = 'data/gastric_cancer.csv'
label = 'Class'

inner_selection = 'one_sem'

# csv_dir = 'data/composite_dataset.csv'
# label = 'group'

mlpipe = MLPipelines(label=label, csv_dir=csv_dir)
df = mlpipe.nested_cv(parallel='freely_parallel',inner_selection=inner_selection)
print(df.head())
