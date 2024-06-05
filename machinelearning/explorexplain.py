import numpy as np
import pandas as pd
# from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display
import ipywidgets as widgets
from .featexpl import FeaturesExplanation
# import shap
from dataloader import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.decomposition import PCA
# import warnings
import umap
# sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})


class ExploreExplain(DataLoader):
    def __init__(self, X, y, normalization_method):
        self.X = X
        self.y = y
        self.normalization_method = normalization_method
        self.x_normalized_df = self.normalize(self.X, method=self.normalization_method)
        
    def correlations(self,data=None,labels=None, list_of_feature=None, limit=None, num_of_best_features=None, way_of_selection='mrmr'):
        """
        Calculates and visualizes the correlation matrix of the dataset, optionally after applying feature selection and normalization.

        This method computes the correlation matrix for the dataset stored in the instance. It supports optional normalization of the data before computing correlations. Additionally, feature selection can be applied to focus the correlation analysis on the most relevant features. The resulting correlation matrix can be filtered to show only correlations that exceed a specified threshold. The method also visualizes the correlation matrix using a heatmap.

        Parameters:
        - data (DataFrame, optional): The input data for PCA. If None, the instance's data is used.
        - labels (Series or array-like, optional): The labels corresponding to the input data. If None, the instance's labels are used.
        - list_of_features (list of str, optional): A list of feature names to include in the correlation analysis. If None, all features in the dataset are used.
        - limit (float, optional): A threshold for filtering the correlations displayed in the heatmap. Only correlations with absolute values greater than or equal to this limit are shown. The value must be between 0 and 1. If None, all correlations are shown.
        - num_of_best_features (int, optional): The number of top features to select for correlation analysis, based on the selection method specified. If None, no feature selection is applied.
        - way_of_selection (str, optional): The method to use for feature selection. Defaults to 'mrmr'. The method must be supported by the feature_selection method of the instance.

        Returns:
        - numpy.ndarray: The correlation matrix of the selected features with or without the specified limit applied. If feature selection or normalization is applied, the matrix corresponds to the processed dataset.

        Raises:
        - Exception: If an unsupported normalization method is specified or if the limit is not between 0 and 1.

        Note:
        - The correlation matrix includes an additional row and column for the label correlations.
        - The heatmap visualization is displayed using the seaborn library, with feature names and labels included for clarity.
        """
        if data is None and labels is None:
            data = self.x_normalized_df
            labels = self.y
        else:
            if self.normalization_method in ['minmax', 'standard']:
                data = self.normalize(data, method=self.normalization_method)
            else: 
                raise Exception("Unsupported normalization method.")
        
        if list_of_feature is not None:
            data_df = data_df[list_of_feature]
            
        if num_of_best_features is not None:
            selected = self.feature_selection(data, labels, method = way_of_selection, num_features = num_of_best_features)
            data = data[selected]
        
        correl_table = np.corrcoef(data, y=labels, rowvar=False)
        
        if limit is not None and 0 < limit < 1:
            mask = np.abs(correl_table) >= limit
            correl_table = np.where(mask, correl_table, np.nan)
        elif limit is not None:
            raise Exception("The limit must be between 0 and 1.")
        
        feature_names = data.columns.to_list()+['labels']
        df_correl = pd.DataFrame(correl_table,index=feature_names, columns=feature_names)
       
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_correl,  cmap='coolwarm')#,annot=True, fmt=".2f",)
        plt.show()
        
        return correl_table
        
        
    def pairplots_function(self,data=None,labels=None,list_of_feature=None, num_of_best_features=10, way_of_selection='mrmr'):
        """
        Generate pairplots to visualize relationships between features and labels.
        
        Parameters:
            - data (DataFrame, optional): The input data for PCA. If None, the instance's data is used.
            - labels (Series or array-like, optional): The labels corresponding to the input data. If None, the instance's labels are used.
            - list_of_feature (list): List of features to include in the pairplots.
            - num_of_best_features (int): Number of best features to select for pairplots.
            - way_of_selection (str): Feature selection method to use.
            
        Returns:
            The pairplot.
        """        
        if data is None and labels is None:
            data = self.x_normalized_df
            labels = self.y
        else:
            if self.normalization_method in ['minmax', 'standard']:
                data = self.normalize(data, method=self.normalization_method)
            else: 
                raise Exception("Unsupported normalization method.")
        
        if list_of_feature is not None:
            data_df = data_df[list_of_feature]
            
        if num_of_best_features is not None:
            selected = self.feature_selection(data, labels, method = way_of_selection, num_features = num_of_best_features)
            data = data[selected]
        
        data['labels'] = labels
                
        # fig = px.scatter_matrix(data, dimensions=data.columns[:-1], color='labels')

        # # Update layout for better readability
        # fig.update_layout(
        #     title='Pairplot of Features',
        #     width=1200,
        #     height=1200
        # )

        # fig.show()
        sns.pairplot(data, hue='labels')
        plt.show()
        
    def statistical_difference(self,data=None,labels=None,p_value=0.05,list_of_feature=None, num_of_best_features=None, way_of_selection='mrmr', normalize=False):
        """
        Perform non-parametric statistical tests to identify significant features based on labels,
        and visualize these features' distributions across groups using boxplots.
        
        This function allows optional data normalization and feature selection before conducting
        the statistical tests. It supports identifying a specified number of best features using
        the provided feature selection method and visualizes the results for features where
        the distributions significantly differ across the groups as determined by the specified p-value.
        
        Parameters:
        - data (DataFrame, optional): The input data for PCA. If None, the instance's data is used.
        - labels (Series or array-like, optional): The labels corresponding to the input data. If None, the instance's labels are used.
        - p_value (float): Significance level for determining statistical significance in tests.
                        Defaults to 0.05.
        - list_of_feature (list, optional): List of features to consider for the statistical tests.
                                            If None, all features in the dataset are used.
                                            Defaults to None.
        - num_of_best_features (int, optional): Number of top features to select based on the specified
                                                feature selection method. If None, no feature selection is applied.
                                                Defaults to None.
        - way_of_selection (str): Feature selection method to use when num_of_best_features is specified.
                                Supported values are 'mrmr' and others as implemented in the feature_selection method.
                                Defaults to 'mrmr'.
        - normalize (boolean, optional): Normalization method to apply to the data before performing
                                                statistical tests. Supported method is 'minmax' .
                                                If None, no normalization is applied. Defaults to None.

        Returns:
        - list: A list of feature names that show statistically significant differences across groups,
                based on the specified p-value threshold. Returns an empty list if no significant features are found.
                
        Raises:
        - Exception: If an unsupported normalization method is specified.
        
        Note:
        No normalization is required for the statistical tests since the methods used are non-parametric.
        """
        if data is None and labels is None:
            data = self.X
            labels = self.y
            if normalize:
                if self.normalization_method == 'minmax':
                    data = self.x_normalized_df
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
            data_df = data_df[list_of_feature]
            
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
            # test_stat, p_value = stats.kruskal(*group_data)
            # Mann-Whitney U test for two groups
            test_stat, p_value = stats.mannwhitneyu(*group_data, alternative='two-sided')
            p_values[feature] = p_value
        
        significant_features = [feature for feature, p in p_values.items() if p < p_value]
        
        if significant_features:
            data_all = data[significant_features].copy()
            data_all['labels'] = labels
            melted_data_all = pd.melt(data_all, id_vars='labels', var_name='variable', value_name='value')            
            print(f'Number of significant features: {len(significant_features)} of {len(data.columns)-1} provided.')
            
            fig = go.Figure()

            for variable in significant_features:
                for label in melted_data_all['labels'].unique():
                    filtered_data = melted_data_all[(melted_data_all['variable'] == variable) & (melted_data_all['labels'] == label)]
                    fig.add_trace(go.Box(
                        y=filtered_data['value'],
                        name=f'{variable} - {label}',
                        jitter=0.3,
                        pointpos=-1.8
                    ))

            fig.update_layout(
                title='Boxplot of Significant Features',
                xaxis_title='Feature',
                yaxis_title='Value',
                template='plotly_white',
                width=max(20, len(significant_features)) * 40,  
                height=400  
            )
            fig.update_xaxes(tickangle=90)
            fig.show()
            return significant_features
        else:
            print("No significant features found.")    
    
    def pca_plot(self,data=None,labels=None,variance_threshold=None,components_resize=None, components_plot=2, missing_values_method='drop'):
        """
        Performs Principal Component Analysis (PCA) and visualizes the results. This function can operate in two main modes:
        1. Variance Threshold Mode: Finds the number of components required to explain a specified threshold of variance.
        2. Components Resize Mode: Directly uses a specified number of components for PCA.

        Parameters:
        - data (DataFrame, optional): The input data for PCA. If None, the instance's data is used.
        - labels (Series or array-like, optional): The labels corresponding to the input data. If None, the instance's labels are used.
        - variance_threshold (float, optional): The cumulative variance threshold for selecting the optimal number of components. Must be between 0 and 1. If specified, the function will find the minimum number of components that cumulatively explain at least this amount of variance.
        - components_resize (int, optional): The exact number of principal components to retain. If specified, `variance_threshold` is ignored, and PCA is performed with this number of components.
        - components_plot (int, default=2): The number of principal components to include in the final scatter plot visualization.

        Returns:
        - numpy.ndarray: The transformed data, reduced to the optimal number of components found based on the variance threshold or specified directly via `components_resize`.

        Raises:
        - Exception: If both `variance_threshold` and `components_resize` are None, indicating that the method of component selection is unspecified.
        - Exception: If `variance_threshold` is specified and is not between 0 and 1.

        Note:
        - This function visualizes the cumulative explained variance ratio by principal components using a line plot, highlighting the selected variance threshold and the corresponding number of components required.
        - A scatter matrix plot is also generated to visualize the relationships between the principal components specified by `components_plot`, colored by the provided labels.
        """
        
        if data is None and labels is None:
            data = self.x_normalized_df
            labels = self.y
        else:
            if self.normalization_method in ['minmax', 'standard']:
                data = self.normalize(data, method=self.normalization_method)
            else: 
                raise Exception("Unsupported normalization method.")
        # data_all = data.copy()
        # data_all['labels'] = labels
        data['labels'] = labels
        
        data = self.missing_values(data, method=missing_values_method)
        data_labels = pd.DataFrame(data['labels'], columns=['labels'])
        data = data.drop(['labels'], axis=1)
                    
        if variance_threshold != None:
            if variance_threshold < 0 or variance_threshold > 1:
                raise Exception("Variance threshold must be between 0 and 1.")
            else: 
                if data.shape[0] < data.shape[1]:
                    # warnings.warn('By default, PCA plot uses n_components == min(n_samples, n_features). ', UserWarning)                    
                    print(f'Warning: By default PCA plot uses n_components == min(n_samples, n_features)\nThus for the following components search the variance threshold will be applied to {data.shape[0]} components.')
                pca = PCA()
                X_pca = pca.fit_transform(data)
                explained_variance_ratio = pca.explained_variance_ratio_
                total_explained_variance_ratio = explained_variance_ratio.sum()            
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
                
        if variance_threshold != None:
            components_resize = components_found
            pca_optimal = PCA(n_components=components_resize)
        elif components_resize == None and variance_threshold == None:
            pca_optimal = PCA()
        elif components_resize != None and variance_threshold == None:
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
        )
        
        fig.update_layout(
            width=900,  
            height=600  
        )
        
        fig.update_traces(diagonal_visible=False)
        fig.show()
        
        return X_pca_optimal
    
    # app.run_server(debug=True)
    
    def umap_plot(self, data=None, labels=None, list_of_feature=None, num_of_best_features=None, way_of_selection='mrmr',
                  missing_values_method='drop', n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=0):
        """
        Perform UMAP dimensionality reduction on the input data and create an interactive plot.
        
        Parameters:
        - data: Input data to be transformed (default: None).
        - labels: Labels corresponding to the data points (default: None).
        - list_of_feature: List of features to consider (default: None).
        - num_of_best_features: Number of best features to select (default: None).
        - way_of_selection: Method for feature selection (default: 'mrmr').
        - missing_values_method: Method to handle missing values (default: 'drop').
        - n_neighbors: Number of neighbors for UMAP (default: 15).
        - min_dist: Minimum distance for UMAP (default: 0.1).
        - n_components: Number of components for UMAP (default: 2).
        - metric: Metric to use for UMAP (default: 'euclidean').
        - random_state: Random state for UMAP (default: 0).
        
        Raises:
        - Exception: If an unsupported normalization method is encountered.
        
        Returns:
        - None
        """
        
        if data is None and labels is None:
            data = self.x_normalized_df
            labels = self.y
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
            marker=dict(color=data['labels'], colorscale='Viridis', showscale=True)
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
        
        
        




# class ExploreExplain(FeaturesExplanation):
#     def __init__(self, best_estimator, X, y, label_mapping, shap_values=None, explainer=None, max_pca=5):
#         super().__init__(best_estimator, X, y, label_mapping)
#         self.explainer = explainer
#         self.name = best_estimator.__class__.__name__
#         self.X = X
#         self.y = y
#         self.shap_values = shap_values
#         self.label_mapping = label_mapping
#         self.max_pca = max_pca
#         self.app = Dash(__name__)
#         self.setup_layout()
#         self.setup_callbacks()



#     def setup_layout(self):
#         self.app.layout = html.Div([
#             html.H4("Visualization of PCA's explained variance"),
#             dcc.Graph(id="graph"),
#             html.P("Number of components:"),
#             dcc.Slider(id='slider', min=2, max=self.max_pca, value=3, step=1),
#         ])

#     def setup_callbacks(self,):
#         @self.app.callback(
#             Output("graph", "figure"), 
#             [Input("slider", "value")]
#         )

#         def run_and_plot(n_components):
#             try:
#                 pca = PCA(n_components=n_components)
#                 components = pca.fit_transform(self.X)

#                 var = pca.explained_variance_ratio_.sum() * 100
#                 labels = {i: f'PC{i+1}' for i in range(n_components)}

#                 fig = px.scatter_matrix(
#                     components,
#                     color=self.y,
#                     dimensions=range(n_components),
#                     labels=labels,
#                     title=f'Total Explained Variance: {var:.2f}%')
#                 fig.update_traces(diagonal_visible=False)
#                 return fig
#             except Exception as e:
#                 return px.scatter_matrix(title=f'An error occurred: {e}')
            
#     def run_server(self):
#             self.app.run_server(debug=True)
    
#     def create_pca_dataframe(self,X, n_components):
#         """
#         Performs PCA on the given dataset X and returns a DataFrame with the principal components.

#         Parameters:
#         - X: array-like, shape (n_samples, n_features)
#             The input data to perform PCA on.
#         - n_components: int
#             The number of principal components to compute.

#         Returns:
#         - df_pca: pandas.DataFrame
#             DataFrame containing the principal components.
#         """
#         pca = PCA(n_components=n_components)
#         components = pca.fit_transform(X)
#         component_names = [f'PC{i+1}' for i in range(n_components)]
#         df_pca = pd.DataFrame(components, columns=component_names)
#         return df_pca

#     def create_interactive_pca_plot(self, n_components, components, image_size=(1000, 500),top_n_features=20):
#         """
#         αυτο το πραμα θεσ να εχει 2 λειτουργιεσ, να παραγει ΠΣΑ πλοτ
#         ειτε για τα σελεκτεντ φιτουρς ειτε για τα σαπ βαλιους.
#         Τωρα κανει το 1ο, φτιαξτο 2ο


#         Creates an interactive plot with PCA on one side and SHAP values on the other.

#         Parameters:
#         - n_components: The number of principal components to compute.
#         - components: Tuple of indices for the PCA components to plot, e.g., (0, 1) for PC1 vs. PC2.
#         - image_size: Tuple for the image size, default to (1000, 500).
#         - top_n_features: Integer, number of top features to show SHAP values for.
#         Returns:
#         - A Plotly FigureWidget with interactive capabilities in a Jupyter environment.
#         """

#         df_pca = self.create_pca_dataframe(self.X, n_components=n_components)
#         pc_x, pc_y = f'PC{components[0]+1}', f'PC{components[1]+1}'
#         sel_features = self.feature_selection(X=self.X, y=self.y, num_features=top_n_features)
#         features_X = self.X[sel_features]
#         mean_values = abs(features_X.values).mean(axis=0)

#         fig = make_subplots(rows=1, cols=2, subplot_titles=('PCA Plot', 'SHAP Values'),
#                             specs=[[{'type': 'scatter'}, {'type': 'bar'}]],
#                             horizontal_spacing=0.1)
        
#         scatter = go.Scatter(x=df_pca[pc_x], y=df_pca[pc_y], mode='markers', name='Data Points',
#                             marker=dict(color=self.y, colorscale='Viridis', showscale=True),
#                             text=self.y)  # Add labels as hover text
#         fig.add_trace(scatter, row=1, col=1)
#         fig.update_xaxes(title_text=pc_x, row=1, col=1)
#         fig.update_yaxes(title_text=pc_y, row=1, col=1)

#         bar = go.Bar(x=[f'Feature {i}' for i in features_X], y=mean_values)
#         fig.add_trace(bar, row=1, col=2)
#         fig.update_xaxes(title_text='Top Features', row=1, col=2)
#         fig.update_yaxes(title_text='Mean |SHAP Value|', row=1, col=2)

#         fig.update_layout(showlegend=False, width=image_size[0], height=image_size[1], clickmode='event+select')

#         fig_widget = go.FigureWidget(fig)

#         def update_plot(trace, points, selector):
#             if points.point_inds:
#                 idx = points.point_inds[0]  
#                 values_point = features_X.values[idx]
#                 with fig_widget.batch_update():
#                     fig_widget.data[1].y = values_point

#         fig_widget.data[0].on_click(update_plot)

#         return fig_widget