import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import os
from typing import Optional, List
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from src.data.process import DataProcessor
from sklearn.decomposition import PCA
import logging


class DataExplorer(DataProcessor):
    def __init__(self,
                label: str,
                csv_dir: str, 
                index_col: Optional[str] = None, 
                normalization: Optional[str] = 'minmax',
                fs_method: Optional[str] = 'mrmr',
                inner_fs_method: Optional[str] = 'chi2',
                mv_method: Optional[str] = 'median',
                class_balance_method: Optional[str] = None
            ) -> None:

        # Call the parent constructor with all the processing parameters
        super().__init__(
            label=label, 
            csv_dir=csv_dir, 
            index_col=index_col,
            normalization=normalization,
            fs_method=fs_method,
            inner_fs_method=inner_fs_method,
            mv_method=mv_method,
            class_balance_method=class_balance_method,  
            preprocess_mode='general'
        )
        
    def plot_preprocess(
            self, 
            features_names: Optional[List[str]] = None, 
            num_features: Optional[int] = None
        ) -> pd.DataFrame:
        """
        Preprocess data for plotting using the functionality from DataProcessor.
        """
        if features_names is None and num_features is None:
            # If no features are specified, use all features
            num_features = self.X.shape[1]
        # Use the process_general method from DataProcessor
        X_processed, y_processed, _ = self.process_general(
            X=self.X,
            y=self.y,
            num_features=num_features,
            features_name_list=features_names
        )
        
        # Create the data frame for visualization
        data = X_processed.copy()
        data["labels"] = y_processed
        return data

    def plot_class_balance(
            self, 
            save_fig: bool = False, 
            title: str = "Class Balance"
        ) -> None:
        """
        Visualize the class balance as a pie chart and print class counts.

        Parameters
        ----------
        save_fig : bool, optional
            Whether to save the pie chart figure. Default is False.
        """
        # Get label counts
        class_counts = pd.Series(self.y).value_counts().sort_index()
        class_names = [self.label_mapping.get(i, str(i)) for i in class_counts.index]

        # Print class counts
        print("Class distribution:")
        for name, count in zip(class_names, class_counts):
            print(f"{name}: {count}")

        # Plot pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            class_counts,
            labels=class_names,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set2", len(class_counts))
        )
        plt.title(title)

        if save_fig:
            save_dir = os.path.join("results", "images", "class_balance")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{title}_class_balance_piechart.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class balance pie chart saved to: {save_path}")

        plt.show()

    def pca_plot(
            self, 
            features_names: Optional[List[str]] = None, 
            num_features: Optional[int] = None
        ) -> None:
        """
        Create a PCA plot to visualize the distribution of samples in the feature space.

        Parameters
        ----------
        features_names : list, optional
            List of feature names to include in the PCA plot. If None, all features are used.
        num_features : int, optional
            Number of features to include in the PCA plot based on feature importance. 
            If None, all features are used.
        """
        
        # Preprocess data
        data_scaled = self.plot_preprocess(
            features_names=features_names,
            num_features= num_features if num_features is not None else self.X.shape[1]
        )
        
        # Perform PCA
        pca = PCA()
        principal_components = pca.fit_transform(data_scaled.drop(columns=['labels']))
        
        # Create a DataFrame for the PCA results
        pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
        pca_df['labels'] = data_scaled['labels'].values
        
        # Plot PCA results
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='labels', data=pca_df)
        plt.title('PCA of Features')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Labels')
        
        # Save the figure
        save_dir = os.path.join("results", "images", "pca")
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, "pca_plot.png")
        plt.savefig(save_path, dpi=300)
        
        logging.info(f"PCA plot saved to: {save_path}")

    def hierarchical_correlation_plot(
        self,
        features_names: list = None,
        num_features: int = None,
        method: str = 'complete',
        figsize: tuple = (50, 50)
    ) -> None:
        """
        Create a hierarchical clustering correlation heatmap for visualizing relationships
        between features in a metabolomics dataset with balanced layout.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Feature dataset containing metabolite measurements. If None, uses the instance's data.
        labels : pandas.Series or array-like, optional
            Binary target labels (0 or 1) for the samples. If None, uses the instance's labels.
        features_names : list, optional
            List of feature names to include in the plot. If provided, overrides num_features.
        num_features : int, optional
            Number of features to include in the plot based on feature importance. 
            If None, all features are used.
        method : str, optional
            Linkage method for hierarchical clustering ('single', 'complete', 'average', 'ward').
            Default is 'complete'.
        figsize : tuple, optional
            Size of the figure (width, height) in inches. Default is (50, 50).
        """
        
        # Set default class names if not provided
        class_names = list(self.label_mapping.values())
        
        # Preprocess data
        data_scaled = self.plot_preprocess(
            features_names=features_names,
            num_features=num_features
        )
        
        # Check if 'labels' is in the columns (it should be after plot_preprocess)
        if 'labels' not in data_scaled.columns:
            raise ValueError("The 'labels' column is missing from the preprocessed data.")
        
        # Extract labels and remove from dataset for correlation calculation
        y = data_scaled['labels'].values
        X = data_scaled.drop(columns=['labels'])
        
        # Calculate feature importance for color coding
        class_0_data = data_scaled[data_scaled['labels'] == 0].drop(columns=['labels'])
        class_1_data = data_scaled[data_scaled['labels'] == 1].drop(columns=['labels'])
        
        if class_0_data.empty or class_1_data.empty:
            raise ValueError("One or both classes have no samples. Cannot compute class differences.")
        
        class_0_mean = class_0_data.mean()
        class_1_mean = class_1_data.mean()
        mean_diff = class_1_mean - class_0_mean  # Preserve sign
        
        # Calculate correlation matrix
        corr = X.corr(method='pearson')
        
        # Create figure with better proportions
        fig = plt.figure(figsize=figsize)
        
        # Adjust the grid to use space more efficiently
        gs = fig.add_gridspec(1, 2, width_ratios=[0.2, 0.8])
        
        # Add axes
        ax1 = fig.add_subplot(gs[0])  # Dendrogram
        ax2 = fig.add_subplot(gs[1])  # Heatmap
        
        # Convert correlation to distance
        distance = 1 - np.abs(corr)
        
        # Compute linkage
        Z = hierarchy.linkage(squareform(distance), method=method)
        
        # Get the ordering of features based on hierarchical clustering
        dendrogram = hierarchy.dendrogram(
            Z, 
            ax=ax1, 
            orientation='left', 
            # no_labels=True,
            leaf_font_size=10,
            color_threshold=0.3 * max(Z[:,2])  # Color threshold for better visualization
        )
        
        # Get the leaf ordering
        index = dendrogram['leaves']
        
        # Reorder the correlation matrix
        corr = corr.iloc[index, index]
        
        # Feature importance for color bar
        importance_series = pd.Series(mean_diff, index=X.columns)
        importance_series = importance_series.iloc[index]
        
        # Plot the heatmap
        sns.heatmap(
            corr, 
            ax=ax2,
            cmap='coolwarm',
            vmin=-1, 
            vmax=1,
            cbar_kws={'label': f'Pearson Correlation'},
            # square=True  # Make cells square for better visualization
        )
        
        # Improve heatmap layout for Y-axis
        ax2.set_yticks(np.arange(len(corr.index)) + 0.5)
        ax2.set_yticklabels(corr.index, fontsize=8)
        ax2.tick_params(axis='y', rotation=0)
        
        # Add X-axis labels (same as Y-axis since correlation matrix is symmetric)
        ax2.set_xticks(np.arange(len(corr.columns)) + 0.5)
        ax2.set_xticklabels(corr.columns, fontsize=8, rotation=90)  # Rotate to avoid overlap
        
        # Create color map for feature importance bar
        cmap_importance = plt.cm.RdBu_r
        norm = plt.Normalize(importance_series.min(), importance_series.max())
        
        # Create feature importance color bar
        feature_colors = np.array([cmap_importance(norm(importance_series[i])) for i in range(len(importance_series))])
        feature_colors = feature_colors.reshape(-1, 1)

        # Clean up dendrogram
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlabel('Distance', fontsize=10)
        
        # Set title
        plt.suptitle(f'Hierarchical Clustering of Feature Correlations\n(Method: {method.capitalize()})', 
                    fontsize=16)
        
        # Create legend for feature importance interpretation
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', 
                    markersize=10, label=f'{class_names[0]}'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', 
                    markersize=10, label=f'{class_names[1]}')
        ]
        
        # Add the legend
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01), 
                ncol=2, frameon=True, fontsize=9)
        
        # Create directory if it doesn't exist
        save_dir = os.path.join("results", "images", "hierarchical")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the figure
        save_path = os.path.join(save_dir, f"hierarchical_correlation_pearson_{method}.png")
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for title and note
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"Hierarchical correlation plot saved to: {save_path}")
        plt.show()
                    
    def box_illustration(
        self,
        features_names: Optional[List[str]] = None,
        num_features: Optional[int] = None,
        dataset_name: str = "dataset"
    ) -> None:
        """
        Generate a boxplot to visualize the distribution of features values to help identify 
        the well fitted normalization method. Saves the plot to results/images/boxplot folder.
        
        :param data: The input data. If None, the instance's data is used. (DataFrame, optional)
        :param labels: The labels corresponding to the input data. If None, the instance's labels are used. (Series or array-like, optional)
        :param features_names: List of features to consider. If None, all features are used. (list, optional)
        :param num_features: Number of top features to select. If None, no feature selection is applied. (int, optional)
        :param dataset_name: Name to use when saving the boxplot file. Defaults to "dataset". (str, optional)
        """
        
        # Create directory if it doesn't exist
        save_dir = os.path.join("results", "images", "boxplot")
        os.makedirs(save_dir, exist_ok=True)
        
        if self.normalization == "minmax":
            data = self.plot_preprocess(
                features_names=features_names,
                num_features=num_features
            )
            
            # Create a DataFrame in long format for seaborn
            melted_data = pd.melt(
                data,
                id_vars="labels",
                value_vars=data.columns[:-1],
                var_name="Feature",
                value_name="Value"
            )
            
            # Calculate an appropriate figure size
            n_features = len(data.columns) - 1  # Exclude 'labels' column
            fig_width = min(n_features * 0.4, 200)  # Cap width at 200 inches for very large feature sets
            
            # Create boxplot with horizontal orientation for better readability with many features
            plt.figure(figsize=(fig_width, 50))
            
            # Create the boxplot
            sns.boxplot(x="Feature", y="Value", data=melted_data)
            plt.title(f"Boxplot of Features - {dataset_name}")
            plt.xticks(rotation=90)
            plt.tight_layout()
            
            # Save the figure
            save_path = os.path.join(save_dir, f"boxplot_of_{dataset_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close the plot to avoid displaying it
            
            print(f"Boxplot saved to: {save_path}")
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization}. Only 'minmax' is supported for boxplot.")
   
    def pairplots_function(
            self,
            features_names: Optional[List[str]] = None,
            num_features: Optional[int] = None
        ) -> None:
            """
            Generate pairplots to visualize relationships between features and labels.

            :param data: The input data for PCA. If None, the instance's data is used.
            :type data: DataFrame, optional
            :param labels: The labels corresponding to the input data. If None, the instance's labels are used.
            :type labels: Series or array-like, optional
            :param list_of_feature: List of features to include in the pairplots.
            :type list_of_feature: list
            :param num_of_best_features: Number of best features to select for pairplots.
            :type num_of_best_features: int
            :param way_of_selection: Feature selection method to use.
            :type way_of_selection: str
            """
            
            data = self.plot_preprocess(
                features_names=features_names,
                num_features=num_features
            )

            sns.pairplot(data, hue="labels")
            plt.show()

    def statistical_difference(
            self,
            features_names: Optional[List[str]] = None,
            num_features: Optional[int] = None,
            stat_test: str = "mannwhitneyu",
            p_value: float = 0.05,
            show_box: bool = True
        ) -> list:
            """
            Perform non-parametric statistical tests to identify significant features based on labels,
            and visualize these features' distributions across groups using horizontal boxplots.

            This function allows optional data normalization and feature selection before conducting
            the statistical tests. It supports identifying a specified number of best features using
            the provided feature selection method and visualizes the results for features where
            the distributions significantly differ across the groups as determined by the specified p-value.

            :param data: The input data for PCA. If None, the instance's data is used. (DataFrame, optional)
            :param labels: The labels corresponding to the input data. If None, the instance's labels are used. (Series or array-like, optional)
            :param p_value: Significance level for determining statistical significance in tests. Defaults to 0.05. (float)
            :param features_names: List of features to consider for the statistical tests.
                                    If None, all features in the dataset are used. Defaults to None. (list, optional)
            :param num_features: Number of top features to select. If None, no feature selection is applied. 
                                Defaults to None. (int, optional)
            :param stat_test: Statistical test to use. Supported values are 'mannwhitneyu' and 'kstest'. 
                            Defaults to 'mannwhitneyu'. (str)
            :param show_box: Whether to show boxplots of significant features. Defaults to True. 
                            It might be timeconsuming for large featured datasets. (boolean)
            :return: A list of feature names that show statistically significant differences across groups,
                    based on the specified p-value threshold. Returns an empty list if no significant features are found. (list)

            :note: No normalization is required for the statistical tests since the methods used are non-parametric.
            """
            
            if stat_test not in ["mannwhitneyu", "kstest"]:
                raise ValueError("Unsupported statistical test. Only 'mannwhitneyu' and 'kstest' are supported.")
            if p_value <= 0 or p_value >= 1:
                raise ValueError("p_value must be between 0 and 1.")
            if self.normalization == 'minmax':
                data = self.plot_preprocess(features_names=features_names, num_features=num_features)
            else:
                raise ValueError(f"Unsupported normalization method: {self.normalization}. Only 'minmax' is supported for statistical tests.")

            p_values = {}
            groups = data["labels"].unique()
            # Set groups ascending
            groups.sort()

            for feature in data.columns[:-1]:  # Excluding 'labels' column
                if stat_test == "mannwhitneyu":
                    # Mann-Whitney U test for two independent samples
                    group1 = data[data["labels"] == groups[0]][feature].values
                    group2 = data[data["labels"] == groups[1]][feature].values
                    _, p = stats.mannwhitneyu(group1, group2, alternative="two-sided")
                    p_values[feature] = p
                elif stat_test == "kstest":
                    # Kolmogorov-Smirnov test for two independent samples
                    group1 = data[data["labels"] == groups[0]][feature].values
                    group2 = data[data["labels"] == groups[1]][feature].values
                    _, p = stats.ks_2samp(group1, group2, alternative="two-sided")
                    p_values[feature] = p
                else:
                    raise ValueError(f"Unsupported statistical test: {stat_test}, only 'mannwhitneyu' and 'kstest' are supported.")

            # Filter the significant features sorted by p-value
            significant_features = [
                feature for feature, p in sorted(p_values.items(), key=lambda item: item[1]) 
                if p < p_value
            ]

            if significant_features:
                if show_box:
                    # Prepare data for plotting
                    data_for_plot = data[significant_features + ["labels"]].copy()
                    
                    # Determine the number of features to plot
                    n_features = len(significant_features)
                    
                    # Create a more suitable layout
                    fig_height = max(6, n_features * 1.2)  # Adjust height based on number of features
                    
                    # Create the figure and axes
                    fig, axes = plt.subplots(figsize=(10, fig_height))
                    
                    # Create a DataFrame in long format for seaborn
                    melted_data = pd.melt(
                        data_for_plot, 
                        id_vars="labels", 
                        value_vars=significant_features,
                        var_name="Feature", 
                        value_name="Value"
                    )
                    
                    # Create horizontal boxplot
                    sns.boxplot(
                        x="Value", 
                        y="Feature", 
                        hue="labels",
                        data=melted_data, 
                        orient="h",
                        palette="Set2"
                    )
                    
                    # Customize the plot
                    plt.title("Statistically Significant Features")
                    plt.xlabel("Value")
                    plt.ylabel("Feature")
                    plt.tight_layout()
                    
                    # Add legend with labels
                    plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    # Show the plot
                    plt.show()
                    
                print(f"Number of significant features: {len(significant_features)} of {len(data.columns)-1} provided.")
                return significant_features
            else:
                print("No significant features found.")
                return []