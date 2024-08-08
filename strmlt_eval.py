import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import ast
import re

def load_excel_to_dataframe(excel_dir, dataset_name):
    # List all files in the excel directory
    files = os.listdir(excel_dir)
    
    # Filter for Excel files that contain the dataset name
    excel_files = [f for f in files if f.endswith('.xlsx') and dataset_name in f]
    
    if not excel_files:
        print(f'No Excel file found containing the dataset name "{dataset_name}" in directory "{excel_dir}".')
        return None
    
    # Assuming there's only one matching file, get the first one
    file_path = os.path.join(excel_dir, excel_files[0])
    
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)
    
    return df

def fix_format(df, metric):
    # Fix string formatting issues in the 'accuracy' column
    def fix_string_list(s):
        s = re.sub(r'\s+', ',', s.strip())  # Replace multiple spaces with a single comma
        s = s.replace('[,', '[').replace(',]', ']')  # Remove commas next to brackets
        s = re.sub(r',+', ',', s)  # Replace multiple commas with a single comma
        return s

    # Check if the metric column is a string or a list and apply appropriate transformation
    if isinstance(df[metric].iloc[0], str):
        df[metric] = df[metric].apply(lambda x: fix_string_list(x) if isinstance(x, str) else x)
        df[metric] = df[metric].apply(ast.literal_eval)
    elif isinstance(df[metric].iloc[0], list):
        pass  # Do nothing if it's already a list
    else:
        raise ValueError("The metric column is neither a string nor a list")
    return df

def plotit(df, metric, one_sel_type):
    # Explode the 'metric' list into separate rows
    df = df.explode(metric)

    # Convert metric column to numeric type
    df[metric] = pd.to_numeric(df[metric])
    
    # Unique model selection types for separate plots
    unique_selections = df['Model_Selection_Type'].unique()

    for selection in unique_selections:
        selected_df = df[df['Model_Selection_Type'] == selection]
        if selection == 'RNCV':
            # 'RNCV' handled like the original first if statement
            selected_df['Estimator_Selection'] = selected_df['Estimator'] + ' (' + selected_df['Inner_Selection'] + ')'
            selected_df = selected_df.sort_values('Estimator_Selection')

            fig = px.box(selected_df, x='Estimator_Selection', y=metric, color='Estimator_Selection')
            fig.update_layout(
                title=f'Boxplot of {metric.capitalize()} for RNCV by Estimator and Inner Selection',
                xaxis_title='Estimator and Inner Selection',
                yaxis_title=metric.capitalize(),
                yaxis=dict(range=[0, 1.1]),
                xaxis=dict(tickangle=-45),
                height=1000,
                width=1500,
                legend_title_text='Estimator Selection'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            # Other types handled with consistent coloring across selections
            selected_df['Estimator_Selection'] = selected_df['Estimator'] + ' (RCV)'
            selected_df = selected_df.sort_values('Estimator_Selection')

            fig = px.box(selected_df, x='Estimator_Selection', y=metric, color='Estimator_Selection')
            fig.update_layout(
                title=f'Boxplot of {metric.capitalize()} for {selection} by Estimator',
                xaxis_title='Estimator and Selection Type',
                yaxis_title=metric.capitalize(),
                yaxis=dict(range=[0, 1.1]),
                xaxis=dict(tickangle=-45),
                height=1000,
                width=1500,
                legend_title_text='Estimator Selection'
            )
            st.plotly_chart(fig, use_container_width=True)
    
def get_filenames_before_extension(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter out only CSV files and extract names before the extension
    file_roots = [os.path.splitext(file)[0] for file in files if file.endswith('.csv')]
    
    return file_roots

def parse_list(s):
    try:
        # Try to parse as a Python list first
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    
    try:
        # Handle space-separated values within brackets
        s = re.sub(r'\s+', ' ', s.strip())
        s = s.replace('[ ', '[').replace(' ]', ']')
        s = s.replace(' ', ',')
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    
    return s
    
def concatenate_csv_files(directory, dataset_name, output_excel):
    methods =  ['gso_1', 'gso_2', 'validation_score', 'one_sem','one_sem_grd']
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter for the relevant CSV files
    csv_files = [f for f in files if f.endswith('.csv') and dataset_name in f]
    
    # Initialize an empty list to hold DataFrames
    df_list = []
    
    # Loop over each CSV file
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(directory, csv_file))

        # Check if 'RCV' is in the file name and assign a model selection type
        if 'RCV' in csv_file:
            df['Model_Selection_Type'] = 'RCV'
        else:
            df['Model_Selection_Type'] = 'RNCV'
        
        # Convert string representations of lists back to actual lists for specified columns
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].str.startswith('[').any():
                df[col] = df[col].apply(parse_list)
        
        # Append the DataFrame to the list
        df_list.append(df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(df_list, ignore_index=True)
    
    # Save the concatenated DataFrame to an Excel file
    concatenated_df.to_excel(output_excel, index=False)
    st.success(f"Excel file '{output_excel}' created successfully.")
    
def create_new_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        st.success(f"Directory '{dir_name}' created successfully. Excel files will be saved at {dir_name}.")
    else:
        st.warning(f"Directory '{dir_name}' already exists.")
    
st.set_page_config(layout="wide")
st.title('Evaluation of ML models on Metabolomics data')
tab_excel, tab_eval = st.tabs(["Excel Maker", "Evaluation"])

# ================== TABS ==================

# Ensure the session state is properly initialized
if 'excel_dir' not in st.session_state:
    st.session_state.excel_dir = None
    
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = None

with st.sidebar:
    current_directories = [f for f in os.listdir('.') if os.path.isdir(f)]
    init_data = current_directories.index('data')
    data_dir = st.selectbox("Select input directory", current_directories, index=init_data,help="Choose the directory that contains the data.")
    
    dataset_names = get_filenames_before_extension(data_dir)
    dataset_name = st.selectbox("Select a dataset", dataset_names)    
    st.session_state.dataset_name = dataset_name
    
with tab_excel:
    st.markdown("# Create EXCEL")
    
    # List current directories and add an option to create a new one
    current_directories = [f for f in os.listdir('.') if os.path.isdir(f)]
    current_directories.append("Add new directory")
    
    scol1, scol2 = st.columns(2)
        
    init_results = current_directories.index('Results')
    results_dir = scol1.selectbox("Select results directory", current_directories,index=init_results,help="Choose the directory that results already exists.")
    
    excel_dir = scol2.selectbox("Select output Excel directory", current_directories,help="Choose the directory to save output Excel.")
    
    # If the user chooses to create a new directory
    if excel_dir == "Add new directory":
        new_dir_name = scol2.text_input("Enter the name of the new directory")
        if scol2.button("Create Directory"):
            create_new_directory(new_dir_name)
            # Update the selectbox to include the newly created directory
            excel_dir = new_dir_name
    
    st.session_state.excel_dir = excel_dir
    
    if st.button("Create Excel"):
        df = concatenate_csv_files(results_dir, st.session_state.dataset_name, f"{excel_dir}/{st.session_state.dataset_name}_concat_results.xlsx")
        if df is not None:
            st.write(df)
            if st.button("Download Excel"):
                st.download_button("Download Excel", df.to_excel(), file_name=f"{st.session_state.dataset_name}.xlsx")

with tab_eval:
    st.markdown("# Evaluation")   
    
    scol1, scol2 = st.columns(2)
    if st.session_state.excel_dir is None:
        current_directories = [f for f in os.listdir('.') if os.path.isdir(f)]        
        excel_dir = scol1.selectbox("Select Excel directory", current_directories)
    
    st.session_state.excel_dir = excel_dir
    
    df = load_excel_to_dataframe(st.session_state.excel_dir, st.session_state.dataset_name)
    
    if df is None:
        st.warning("No Excel file found.")
        st.stop()
    
    estimators = df['Estimator'].unique()
    selected_estimators = scol1.multiselect("Select Estimators", estimators, key="select_estimators")
    if selected_estimators:
        eval_df = df[df['Estimator'].isin(selected_estimators)]
    else:
        eval_df = df.copy(deep=True)
    
    features = df['Numbers_of_Features'].unique()
    selected_features = scol1.multiselect("Select Number of Features", features, key="select_features")
    
    if selected_features:
        eval_df = eval_df[eval_df['Numbers_of_Features'].isin(selected_features)]

    types = df['Model_Selection_Type'].unique()
    ms_type = scol2.multiselect("Select model selection type", types, key="ms_type", help="Choose between RCV and RNCV.")
    if ms_type:
        eval_df = eval_df[eval_df['Model_Selection_Type'].isin(ms_type)]
    
    one_sel_type = True
    if len(eval_df['Model_Selection_Type'].unique())>1:
        one_sel_type = False

    exclude_columns = ['Estimator', 'Classifier', 'Max', 'Std', 'SEM', 'Median','Hyperparameters','Selected_Features','Inner_Selection','Samples_classification_rates','Numbers_of_Features','Way_of_Selection','Model_Selection_Type']
    available_columns = [col for col in eval_df.columns if col not in exclude_columns]
    column_eval = scol2.selectbox("Select a column to evaluate", available_columns, help="Choose the metric to evaluate.")

    if st.button("Debug"):
        st.dataframe(eval_df)
        
    if st.button("Evaluate"):
        if eval_df is not None:
            eval_df = fix_format(eval_df, column_eval)
            plotit(eval_df, column_eval, one_sel_type)
            
    
