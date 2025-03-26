import pandas as pd
import os
import re

# Select the dataset name
dataset = 'ICC'

# List of the inner selection methods
inner_selection_list = ['gso_1', 'gso_2', 'one_sem_grd', 'one_sem', 'validation_score']

# Get the list of all the files in the results/csv folder
files = os.listdir('results/csv')
# Filter the files to only get the .csv files
files = [file for file in files if file.endswith('.csv')]
filtered_files = []
# Keep only the files that contain the dataset name
for file in files:
    if dataset in file:
        filtered_files.append(file)

# Create a dictionary to store the results
summary_results = pd.DataFrame()

# Using the filtered files names find the number of features
for file in filtered_files:
    # Extract the number of features from the file name which is the number after the features_name_ in the file name
    num_features = re.findall(r'features_name_(\d+)_', file)[0]
    try:
        num_features = int(num_features)
    except:
        continue
    # Extract the estimator name from the file name which is the name after the estimator_name in the file name
    estimator_name = re.findall(r'estimator_name_(.*?)_', file)[0]

    # Extract the inner selection from the file name which is the name after the inner_selection in the file name
    for inner in inner_selection_list:
        if inner in file:
            inner_selection = inner
            break

    # Read the csv file
    df = pd.read_csv(f'results/csv/{file}')
    # Add a column with the number of features
    df['num_features'] = num_features
    # Add a column with the estimator name
    df['estimator_name'] = estimator_name
    # Add a column with the inner selection name
    df['inner_selection'] = inner_selection

    # Add the dataframe to the dictionary
    summary_results = pd.concat([summary_results, df])

# Save the results to a csv file
summary_results.to_csv(f'results/csv/{dataset}_summary_results.csv', index=False)