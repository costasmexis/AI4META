import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
import os

st.set_page_config(layout="wide")
st.title('Papers  Replication')
tab_papers, = st.tabs(["Papers Results"])

with st.sidebar:
    st.markdown("# Select Folder")

    uploadedFile = st.file_uploader("Choose a file",type=['csv','xlsx'],accept_multiple_files=False,key="uploadedFile")

    if uploadedFile is not None:
        st.success("File uploaded successfully")

# Access the correct tab using the unpacked tab_papers
with tab_papers:
    st.markdown("# Papers Results")

    if uploadedFile is not None:
        df = pd.read_csv(uploadedFile)

        # Extract only the names between 'data/' and '.csv'
        df['Dataset Name'] = df['Dataset'].apply(lambda x: x.split('/')[1].split('.')[0])
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        # Selection boxes
        select_dataset = col1.selectbox("Select Dataset", df['Dataset Name'].unique())
        select_metric = col2.selectbox("Select Metric", ['matthews_corrcoef', 'roc_auc', 'f1', 'accuracy', 'balanced_accuracy', 'precision', 'recall'])

        select_tr_methods = col1.multiselect("Select Training Methods", df['Training Method'].unique())
        select_estimators = col2.multiselect("Select Estimators", df['Estimator'].unique())
        select_features = col1.multiselect("Select set of Features", df['Features'].unique())

        # Set default metric if none selected (though selectbox doesn't allow none)
        if select_metric is None:
            select_metric = 'matthews_corrcoef'
        
        # Check if all selections are made
        if (select_dataset) and (select_metric) :

            # Filter the dataframe based on selections
            temp_df = df[
                (df['Dataset Name'] == select_dataset) 
            ]

            if (select_estimators == []) or (select_estimators is None):
                select_estimators = temp_df['Estimator'].unique()
            if (select_tr_methods == []) or (select_tr_methods is None):
                select_tr_methods = temp_df['Training Method'].unique()
            if (select_features == []) or (select_features is None):
                select_features = temp_df['Features'].unique()

            temp_df = temp_df[
                (temp_df['Estimator'].isin(select_estimators)) &
                (temp_df['Training Method'].isin(select_tr_methods)) &
                (temp_df['Features'].isin(select_features))
            ]
            
            # Create 'Estimator_groups' based on 'Features'
            def create_estimator_groups(row):
                if row['Features'] == 'all':
                    return f"{row['Estimator']}_all"
                else:
                    # Convert string representation of list back to a list
                    feature_list = ast.literal_eval(row['Features'])
                    return f"{row['Estimator']}_{len(feature_list)}"

            temp_df['Estimator_groups'] = temp_df.apply(create_estimator_groups, axis=1)  

        if not temp_df.empty:
            for method in select_tr_methods:
                method_df = temp_df[temp_df['Training Method'] == method]

                # Create a box plot
                fig = px.box(
                    method_df, 
                    x='Estimator_groups', 
                    y=select_metric, 
                    color='Estimator_groups',
                    title=f'Boxplot of {select_metric.capitalize()} for {select_dataset} - {method}',
                    labels={select_metric: select_metric.capitalize(), 'Estimator_groups': 'Estimator_groups'}
                )

                # Calculate mean for each estimator
                means = method_df.groupby('Estimator_groups')[select_metric].mean().reset_index()

                # Add mean lines to the box plot
                for estimator in means['Estimator_groups']:
                    mean_value = means[means['Estimator_groups'] == estimator][select_metric].values[0]
                    
                    fig.add_shape(
                        type='line',
                        x0=estimator,
                        x1=estimator,
                        y0=mean_value,
                        y1=mean_value,
                        line=dict(color='Red', width=2, dash='dash'),
                        xref='x',
                        yref='y'
                    )

                    # Add text annotation next to the line indicating the mean value
                    fig.add_annotation(
                        x=estimator,
                        y=mean_value,
                        text=f"Mean: {mean_value:.2f}",
                        showarrow=False,
                        yshift=10
                    )

                fig.update_layout(
                    yaxis=dict(range=[0, 1.1]),
                    xaxis=dict(tickangle=-45),
                    height=600,
                    width=1000,
                    legend_title_text='Estimator_groups'
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
        st.warning("There might be estimators that are not available in the dataset.")
    else:
        st.stop()
