import pandas as pd 

dataframe_1 = 'results/ICC/ICC__features_name_[10]_20241230_2204_rncv_outerloops_results.csv'
dataframe_2 = 'results/ICC/ICC__features_name_[10]_20241230_2247_rcv_outerloops_results.csv'

df_1 = pd.read_csv(dataframe_1)
df_2 = pd.read_csv(dataframe_2)

merged = pd.concat([df_1, df_2], ignore_index=True)
merged.to_excel('results/ICC/ICC_features_name_[10]_rncv_rcv_merged.xlsx')
print('FINISHED')