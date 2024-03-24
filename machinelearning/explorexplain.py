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

class ExploreExplain(FeaturesExplanation):
    def __init__(self, best_estimator, X, y, label_mapping, shap_values=None, explainer=None, max_pca=5):
        super().__init__(best_estimator, X, y, label_mapping)
        self.explainer = explainer
        self.name = best_estimator.__class__.__name__
        self.X = X
        self.y = y
        self.shap_values = shap_values
        self.label_mapping = label_mapping
        self.max_pca = max_pca
        self.app = Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()



    def setup_layout(self):
        self.app.layout = html.Div([
            html.H4("Visualization of PCA's explained variance"),
            dcc.Graph(id="graph"),
            html.P("Number of components:"),
            dcc.Slider(id='slider', min=2, max=self.max_pca, value=3, step=1),
        ])

    def setup_callbacks(self,):
        @self.app.callback(
            Output("graph", "figure"), 
            [Input("slider", "value")]
        )

        def run_and_plot(n_components):
            try:
                pca = PCA(n_components=n_components)
                components = pca.fit_transform(self.X)

                var = pca.explained_variance_ratio_.sum() * 100
                labels = {i: f'PC{i+1}' for i in range(n_components)}

                fig = px.scatter_matrix(
                    components,
                    color=self.y,
                    dimensions=range(n_components),
                    labels=labels,
                    title=f'Total Explained Variance: {var:.2f}%')
                fig.update_traces(diagonal_visible=False)
                return fig
            except Exception as e:
                return px.scatter_matrix(title=f'An error occurred: {e}')
            
    def run_server(self):
            self.app.run_server(debug=True)
    
    def create_pca_dataframe(self,X, n_components):
        """
        Performs PCA on the given dataset X and returns a DataFrame with the principal components.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            The input data to perform PCA on.
        - n_components: int
            The number of principal components to compute.

        Returns:
        - df_pca: pandas.DataFrame
            DataFrame containing the principal components.
        """
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X)
        component_names = [f'PC{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(components, columns=component_names)
        return df_pca

    def create_interactive_pca_plot(self, n_components, components, image_size=(1000, 500),top_n_features=20):
        """
        αυτο το πραμα θεσ να εχει 2 λειτουργιεσ, να παραγει ΠΣΑ πλοτ
        ειτε για τα σελεκτεντ φιτουρς ειτε για τα σαπ βαλιους.
        Τωρα κανει το 1ο, φτιαξτο 2ο


        Creates an interactive plot with PCA on one side and SHAP values on the other.

        Parameters:
        - n_components: The number of principal components to compute.
        - components: Tuple of indices for the PCA components to plot, e.g., (0, 1) for PC1 vs. PC2.
        - image_size: Tuple for the image size, default to (1000, 500).
        - top_n_features: Integer, number of top features to show SHAP values for.
        Returns:
        - A Plotly FigureWidget with interactive capabilities in a Jupyter environment.
        """

        df_pca = self.create_pca_dataframe(self.X, n_components=n_components)
        pc_x, pc_y = f'PC{components[0]+1}', f'PC{components[1]+1}'
        sel_features = self.feature_selection(X=self.X, y=self.y, num_features=top_n_features)
        features_X = self.X[sel_features]
        mean_values = abs(features_X.values).mean(axis=0)

        fig = make_subplots(rows=1, cols=2, subplot_titles=('PCA Plot', 'SHAP Values'),
                            specs=[[{'type': 'scatter'}, {'type': 'bar'}]],
                            horizontal_spacing=0.1)
        
        scatter = go.Scatter(x=df_pca[pc_x], y=df_pca[pc_y], mode='markers', name='Data Points',
                            marker=dict(color=self.y, colorscale='Viridis', showscale=True),
                            text=self.y)  # Add labels as hover text
        fig.add_trace(scatter, row=1, col=1)
        fig.update_xaxes(title_text=pc_x, row=1, col=1)
        fig.update_yaxes(title_text=pc_y, row=1, col=1)

        bar = go.Bar(x=[f'Feature {i}' for i in features_X], y=mean_values)
        fig.add_trace(bar, row=1, col=2)
        fig.update_xaxes(title_text='Top Features', row=1, col=2)
        fig.update_yaxes(title_text='Mean |SHAP Value|', row=1, col=2)

        fig.update_layout(showlegend=False, width=image_size[0], height=image_size[1], clickmode='event+select')

        fig_widget = go.FigureWidget(fig)

        def update_plot(trace, points, selector):
            if points.point_inds:
                idx = points.point_inds[0]  
                values_point = features_X.values[idx]
                with fig_widget.batch_update():
                    fig_widget.data[1].y = values_point

        fig_widget.data[0].on_click(update_plot)

        return fig_widget