import pandas as pd 

dataframe_1 = 'results/epic_lc_ms_pos/epic_lc_ms_pos__features_name_[10]_20241229_1935_rncv_outerloops_results.csv'
dataframe_2 = 'results/epic_lc_ms_pos/epic_lc_ms_pos__features_name_[10]_20241229_1937_rcv_outerloops_results.csv'

df_1 = pd.read_csv(dataframe_1)
df_2 = pd.read_csv(dataframe_2)

merged = pd.concat([df_1, df_2], ignore_index=True)
merged.to_excel('results/epic_lc_ms_pos/epic_lc_ms_pos_features_name_[10]_rncv_rcv_merged.xlsx')