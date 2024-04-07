import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display
import ipywidgets as widgets
from .featexpl import FeaturesExplanation
import shap
from dataloader import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt



class ExploreExplain(DataLoader):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def correlations(self, list_of_feature=None, limit=None, num_of_best_features=None, way_of_selection='mrmr', normalization_method=None):
        """
        Calculates and visualizes the correlation matrix of the dataset, optionally after applying feature selection and normalization.

        This method computes the correlation matrix for the dataset stored in the instance. It supports optional normalization of the data before computing correlations. Additionally, feature selection can be applied to focus the correlation analysis on the most relevant features. The resulting correlation matrix can be filtered to show only correlations that exceed a specified threshold. The method also visualizes the correlation matrix using a heatmap.

        Parameters:
        - list_of_features (list of str, optional): A list of feature names to include in the correlation analysis. If None, all features in the dataset are used.
        - limit (float, optional): A threshold for filtering the correlations displayed in the heatmap. Only correlations with absolute values greater than or equal to this limit are shown. The value must be between 0 and 1. If None, all correlations are shown.
        - num_of_best_features (int, optional): The number of top features to select for correlation analysis, based on the selection method specified. If None, no feature selection is applied.
        - way_of_selection (str, optional): The method to use for feature selection. Defaults to 'mrmr'. The method must be supported by the feature_selection method of the instance.
        - normalization_method (str, optional): The method to use for data normalization before computing correlations. Supported values are 'minmax' and 'standard'. If None, no normalization is applied.

        Returns:
        - numpy.ndarray: The correlation matrix of the selected features with or without the specified limit applied. If feature selection or normalization is applied, the matrix corresponds to the processed dataset.

        Raises:
        - Exception: If an unsupported normalization method is specified or if the limit is not between 0 and 1.

        Note:
        - The correlation matrix includes an additional row and column for the label correlations.
        - The heatmap visualization is displayed using the seaborn library, with feature names and labels included for clarity.
        """
        data = self.X
        labels = self.y
       
        if normalization_method != None:
            if normalization_method in ['minmax', 'standard']:
                data = self.normalize(data, method=normalization_method)
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