import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import seaborn as sns

import umap
from dash import Dash, Input, Output, dcc, html
from IPython.display import display
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from .dataloader import DataLoader

class DataExplorer(DataLoader):
    def __init__(self, label, csv_dir, normalization_method: str = 'minmax'):
        super().__init__(label, csv_dir)
        self.normalization_method = normalization_method
        self.X_normalized_df = self.normalize(X=self.X, method=self.normalization_method)
    
    # TODO: Maybe use also seaborn.clustermap for better visualization
    def correlations(self, data=None, labels=None, list_of_feature=None, 
                     limit=None, num_of_best_features=None, way_of_selection='mrmr', 
                     get_table: bool = False):
        """
        Calculates and visualizes the correlation matrix of the dataset, optionally after applying feature selection and normalization.

        This method computes the correlation matrix for the dataset stored in the instance. It supports optional normalization of the data before computing correlations. Additionally, feature selection can be applied to focus the correlation analysis on the most relevant features. The resulting correlation matrix can be filtered to show only correlations that exceed a specified threshold. The method also visualizes the correlation matrix using a heatmap.

        :param data: The input data for PCA. If None, the instance's data is used. (DataFrame, optional)
        :param labels: The labels corresponding to the input data. If None, the instance's labels are used. (Series or array-like, optional)
        :param list_of_features: A list of feature names to include in the correlation analysis. If None, all features in the dataset are used. (list of str, optional)
        :param limit: A threshold for filtering the correlations displayed in the heatmap. Only correlations with absolute values greater than or equal to this limit are shown. The value must be between 0 and 1. If None, all correlations are shown. (float, optional)
        :param num_of_best_features: The number of top features to select for correlation analysis, based on the selection method specified. If None, no feature selection is applied. (int, optional)
        :param way_of_selection: The method to use for feature selection. Defaults to 'mrmr'. The method must be supported by the feature_selection method of the instance. (str, optional)
        :param get_table: A boolean flag to return the correlation matrix as a numpy array. If True, the method returns the correlation matrix. If False, the method only visualizes the heatmap. (bool, optional)
        
        :return: The correlation matrix of the selected features with or without the specified limit applied. If feature selection or normalization is applied, the matrix corresponds to the processed dataset. (numpy.ndarray)

        :raises Exception: If an unsupported normalization method is specified or if the limit is not between 0 and 1.

        :note: The correlation matrix includes an additional row and column for the label correlations.
        :note: The heatmap visualization is displayed using the seaborn library, with feature names and labels included for clarity.
        """
        if data is None and labels is None:
            data = self.X_normalized_df
            labels = self.y
        else:
            if self.normalization_method in ['minmax', 'standard']:
                data = self.normalize(data, method=self.normalization_method)
            else: 
                raise Exception("Unsupported normalization method.")

        
        if list_of_feature is not None:
            data = data[list_of_feature]
            
        if num_of_best_features is not None:
            selected = self.feature_selection(data, labels, method = way_of_selection, num_features = num_of_best_features)
            data = data[selected]
        
        # Drop rows with missing values to generate correlation table
        data = self.missing_values(data=data, method='drop', verbose=False)
        labels = pd.DataFrame(self.y).iloc[data.index][0].values

        correl_table = np.corrcoef(data, y=labels, rowvar=False)
        
        if limit is not None and 0 < limit < 1:
            mask = np.abs(correl_table) >= limit
            correl_table = np.where(mask, correl_table, np.nan)
        elif limit is not None:
            raise Exception("The limit must be between 0 and 1.")
        
        feature_names = data.columns.to_list()+['labels']
        df_correl = pd.DataFrame(correl_table,index=feature_names, columns=feature_names)
                
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_correl, annot=True, fmt=".3f", cmap='crest')
        plt.show()

        if get_table:
            return correl_table
            
    def pairplots_function(self, data=None, labels=None, list_of_feature=None, 
                               num_of_best_features=10, way_of_selection='mrmr') -> None:
            """
            Generate pair plots for visualizing relationships between features.

            :param data: The input data, defaults to None.
            :type data: pandas.DataFrame, optional
            :param labels: The labels for the data, defaults to None.
            :type labels: pandas.Series, optional
            :param list_of_feature: A list of features to include in the pair plots, defaults to None.
            :type list_of_feature: list, optional
            :param num_of_best_features: The number of best features to select, defaults to 10.
            :type num_of_best_features: int, optional
            :param way_of_selection: The method for feature selection, defaults to 'mrmr'.
            :type way_of_selection: str, optional
            """
            if data is None and labels is None:
                data = self.X_normalized_df
                labels = self.y
            else:
                if self.normalization_method in ['minmax', 'standard']:
                    data = self.normalize(data, method=self.normalization_method)
                else: 
                    raise Exception("Unsupported normalization method.")
            
            if list_of_feature is not None:
                data = data[list_of_feature]
                
            if num_of_best_features is not None:
                selected = self.feature_selection(data, labels, method=way_of_selection, num_features=num_of_best_features)
                data = data[selected]
            
            # Drop rows with missing values to generate correlation table
            data = self.missing_values(data=data, method='drop', verbose=False)
            labels = pd.DataFrame(self.y).iloc[data.index][0].values

            data['labels'] = labels
            sns.pairplot(data, hue='labels')
            plt.show()
        
    def statistical_difference(self, data=None, labels=None, p_value=0.05, list_of_feature=None, 
                               num_of_best_features=None, way_of_selection='mrmr', normalize=False) -> list:
        """
        Perform non-parametric statistical tests to identify significant features based on labels,
        and visualize these features' distributions across groups using boxplots.
        
        This function allows optional data normalization and feature selection before conducting
        the statistical tests. It supports identifying a specified number of best features using
        the provided feature selection method and visualizes the results for features where
        the distributions significantly differ across the groups as determined by the specified p-value.
        
        :param data: The input data for PCA. If None, the instance's data is used.
        :type data: pandas.DataFrame, optional
        :param labels: The labels corresponding to the input data. If None, the instance's labels are used.
        :type labels: pandas.Series or array-like, optional
        :param p_value: Significance level for determining statistical significance in tests. Defaults to 0.05.
        :type p_value: float
        :param list_of_feature: List of features to consider for the statistical tests. If None, all features in the dataset are used. Defaults to None.
        :type list_of_feature: list, optional
        :param num_of_best_features: Number of top features to select based on the specified feature selection method. If None, no feature selection is applied. Defaults to None.
        :type num_of_best_features: int, optional
        :param way_of_selection: Feature selection method to use when num_of_best_features is specified. Supported values are 'mrmr' and others as implemented in the feature_selection method. Defaults to 'mrmr'.
        :type way_of_selection: str
        :param normalize: Normalization method to apply to the data before performing statistical tests. Supported method is 'minmax'. If None, no normalization is applied. Defaults to None.
        :type normalize: boolean, optional
        
        :return: A list of feature names that show statistically significant differences across groups, based on the specified p-value threshold. Returns an empty list if no significant features are found.
        :rtype: list
        
        :note: No normalization is required for the statistical tests since the methods used are non-parametric.
        """
        if data is None and labels is None:
            data = self.X
            labels = self.y
            if normalize:
                if self.normalization_method == 'minmax':
                    data = self.X_normalized_df
                else: 
                    print("WARNING: Unsupported normalization method. For non-parametric tests, only 'minmax' is supported. The normalization will automaticly change to 'minmax' for this function only.")
                    data = self.normalize(self.X, method='minmax')
        else:
            if normalize:
                if self.normalization_method in ['minmax']:
                    data = self.normalize(data, method=self.normalization_method)
                else: 
                    raise Exception("Unsupported normalization method. For non-parametric tests, only 'minmax' is supported.")

        if list_of_feature is not None:
            data = data[list_of_feature]
            
        if num_of_best_features is not None:
            selected = self.feature_selection(data, labels, method = way_of_selection, num_features = num_of_best_features)
            data = data[selected]
        
        data['labels'] = labels
        
        p_values = {}
        
        groups = data['labels'].unique()
        
        for feature in data.columns[:-1]:
            group_data = [data[feature][labels == group] for group in groups]
            # Perform the appropriate statistical test based on the number of groups
            # Kruskal-Wallis H-test for more than two groups
            # Mann-Whitney U test for two groups
            _, p_value = stats.mannwhitneyu(*group_data, alternative='two-sided')
            p_values[feature] = p_value
        
        significant_features = [feature for feature, p in p_values.items() if p < p_value]
        
        if significant_features:
            data_all = data[significant_features].copy()
            data_all['labels'] = labels
            melted_data_all = pd.melt(data_all, id_vars='labels', var_name='variable', value_name='value')            
            print(f'Number of significant features: {len(significant_features)} of {len(data.columns)-1} provided.')
            
            plt.figure(figsize=(max(12, len(significant_features)), 10))            
            sns.boxplot(x='variable', y='value', hue='labels', data=melted_data_all) 
            plt.xticks(rotation=90)
            plt.title('Boxplot of Significant Features')
            plt.tight_layout()
            plt.show()
            return significant_features
        else:
            print("No significant features found.")
    
    def pca_plot(self, data=None, labels=None, variance_threshold=None, components_resize=None, components_plot=2, missing_values_method='drop'):
        """Perform PCA analysis and plot the results.

        :param data: The input data for PCA. If None, the instance's data is used. Defaults to None.
        :type data: pandas.DataFrame, optional
        :param labels: The labels corresponding to the input data. If None, the instance's labels are used. Defaults to None.
        :type labels: pandas.Series or array-like, optional
        :param variance_threshold: The variance threshold used to determine the number of principal components to keep. If None, all components are kept. Defaults to None.
        :type variance_threshold: float, optional
        :param components_resize: The number of principal components to resize the data to. If None, no resizing is performed. Defaults to None.
        :type components_resize: int, optional
        :param components_plot: The number of principal components to plot. Defaults to 2.
        :type components_plot: int, optional
        :param missing_values_method: The method to handle missing values in the data. Defaults to 'drop'.
        :type missing_values_method: str, optional
        :return: The transformed data after PCA analysis.
        :rtype: numpy.ndarray
        """
        
        if data is None and labels is None:
            data = self.X_normalized_df.copy() # Use .copy() to avoid overwriting the original data
            labels = self.y.copy()
        else:
            if self.normalization_method in ['minmax', 'standard']:
                data = self.normalize(data, method=self.normalization_method)
            else: 
                raise Exception("Unsupported normalization method.")

        data['labels'] = labels
        
        data = self.missing_values(data, method=missing_values_method)
        data_labels = pd.DataFrame(data['labels'], columns=['labels'])
        data = data.drop(['labels'], axis=1)
                    
        if variance_threshold is not None:
            if variance_threshold < 0 or variance_threshold > 1:
                raise Exception("Variance threshold must be between 0 and 1.")
            else: 
                if data.shape[0] < data.shape[1]:
                    print(f'Warning: By default PCA plot uses n_components == min(n_samples, n_features)\nThus for the following components search the variance threshold will be applied to {data.shape[0]} components.')
                pca = PCA()
                pca.fit_transform(data)
                explained_variance_ratio = pca.explained_variance_ratio_
                cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
                components_found = np.where(cumulative_variance_ratio >= variance_threshold)[0][0] + 1
                
                hover_text = [f"Component: {i+1}<br>Individual Variance: {var:.3%}<br>Cumulative Variance: {cum_var:.3%}" 
                            for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio))]

                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.arange(1, len(cumulative_variance_ratio)+1), y=cumulative_variance_ratio,
                            mode='lines+markers', name='Cumulative Variance',
                            hoverinfo='text', text=hover_text,
                            marker=dict(color='RoyalBlue', size=8), line=dict(width=2)))
                
                fig.add_hline(y=variance_threshold, line_dash="dash", line_color="red",
                            annotation_text=f"{variance_threshold*100}% Variance Threshold",
                            annotation_position="bottom right")
                
                fig.add_vline(x=components_found, line_dash="dash", line_color="green",
                            annotation_text=f"{components_found} Components", annotation_position="top left")
                
                fig.update_layout(title='PCA - Cumulative and Individual Variance Explained',
                                xaxis_title='Number of Principal Components',
                                yaxis_title='Variance Explained',
                                hovermode='closest', template='plotly_white')
                
                fig.show()
                
        if variance_threshold is not None:
            components_resize = components_found
            pca_optimal = PCA(n_components=components_resize)
        elif components_resize is None and variance_threshold == None:
            pca_optimal = PCA()
        elif components_resize is not None and variance_threshold is None:
            pca_optimal = PCA(n_components=components_resize)
        
        X_pca_optimal = pca_optimal.fit_transform(data)
        
        total_var = pca_optimal.explained_variance_ratio_.sum() * 100
        lab = {
                str(i): f"PC{i+1}({var:.1f}%)"
                for i, var in enumerate(pca_optimal.explained_variance_ratio_ * 100)
                }

        fig = px.scatter_matrix(
            X_pca_optimal,
            color=data_labels.labels,
            dimensions=range(components_plot),
            labels=lab,
            title=f'Total Explained Variance: {total_var:.2f}%',
            color_continuous_scale=px.colors.diverging.Portland
        )
        
        fig.update_layout(
            width=900,  
            height=600  
        )
        
        fig.update_traces(diagonal_visible=False)
        fig.show()
        
        return X_pca_optimal
    
    def umap_plot(self, data=None, labels=None, list_of_feature=None, num_of_best_features=None, way_of_selection='mrmr',
                  missing_values_method='drop', n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=0):
        """
        Perform UMAP dimensionality reduction on the input data and create an interactive plot.

        :param data: Input data to be transformed (default: None).
        :type data: pandas.DataFrame, optional
        :param labels: Labels corresponding to the data points (default: None).
        :type labels: pandas.Series or array-like, optional
        :param list_of_feature: List of features to consider (default: None).
        :type list_of_feature: list, optional
        :param num_of_best_features: Number of best features to select (default: None).
        :type num_of_best_features: int, optional
        :param way_of_selection: Method for feature selection (default: 'mrmr').
        :type way_of_selection: str, optional
        :param missing_values_method: Method to handle missing values (default: 'drop').
        :type missing_values_method: str, optional
        :param n_neighbors: Number of neighbors for UMAP (default: 15).
        :type n_neighbors: int, optional
        :param min_dist: Minimum distance for UMAP (default: 0.1).
        :type min_dist: float, optional
        :param n_components: Number of components for UMAP (default: 2).
        :type n_components: int, optional
        :param metric: Metric to use for UMAP (default: 'euclidean').
        :type metric: str, optional
        :param random_state: Random state for UMAP (default: 0).
        :type random_state: int, optional

        :raises Exception: If an unsupported normalization method is encountered.

        :return: None
        :rtype: None
        """
        
        if data is None and labels is None:
            data = self.X_normalized_df.copy() # Use .copy() to avoid overwriting the original data
            labels = self.y.copy()
        else:
            if self.normalization_method in ['minmax', 'standard']:
                data = self.normalize(data, method=self.normalization_method)
            else: 
                raise Exception("Unsupported normalization method.")
            
        data_all = data.copy()
        data_all['labels'] = labels
        
        data_all = self.missing_values(data_all, method=missing_values_method)
        data = data_all.drop(['labels'], axis=1)
        
        if list_of_feature is not None:
            data = data[list_of_feature]
            
        if num_of_best_features is not None:
            selected = self.feature_selection(data, labels, method = way_of_selection, num_features = num_of_best_features)
            data = data[selected]
            
        data['labels'] = data_all['labels']
        
        if all(not isinstance(param, list) for param in [n_neighbors, min_dist, n_components, metric, random_state]):
            if n_components == 1:
                print('Warning: n_components == 1. No UMAP projection will be performed with n_components = 2.')
                n_components = 2
            
            reducer = umap.UMAP(
                n_neighbors=n_neighbors, 
                min_dist=min_dist, 
                n_components=n_components, 
                metric=metric, 
                random_state=random_state
            )
            
            data_less = reducer.fit_transform(data)
            fig = go.Figure(data=[go.Scatter(
            x=data_less[:, 0], 
            y=data_less[:, 1], 
            mode='markers',
            marker=dict(color=data['labels'], colorscale=px.colors.diverging.Portland, showscale=True)
        )])
            fig.update_layout(
                    width=900,  
                    height=700  
            )
            fig.update_layout(title='UMAP Projection colored by labels', xaxis_title='UMAP 1', yaxis_title='UMAP 2')
            fig.show()
        else: 
            from itertools import product
            param_list = [n_neighbors, min_dist, n_components, metric, random_state]
            for combo in product(*(param if isinstance(param, list) else [param] for param in param_list)):
                n_neighbors, min_dist, n_components, metric, random_state = combo
                print(f"Testing combination: n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}, metric={metric}, random_state={random_state}")
                
                if n_components == 1:
                    print('Warning: n_components == 1. No UMAP projection will be performed with n_components = 2.')
                    n_components = 2
                
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors, 
                    min_dist=min_dist, 
                    n_components=n_components, 
                    metric=metric, 
                    random_state=random_state
                )
                
                data_less = reducer.fit_transform(data)
                
                fig = go.Figure(data=[go.Scatter(
                x=data_less[:, 0], 
                y=data_less[:, 1], 
                mode='markers',
                marker=dict(color=data['labels'], colorscale='Viridis', showscale=True)
                )])
                fig.update_layout(
                    width=900,  
                    height=700  
                )
                fig.update_layout(title=f'UMAP for combo: n_neighbors={n_neighbors}, min_dist={min_dist},n_components={n_components}, metric={metric}, random_state={random_state}', xaxis_title='UMAP 1', yaxis_title='UMAP 2')
                fig.show()