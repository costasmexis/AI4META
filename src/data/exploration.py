import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import os

from src.data.dataloader import DataLoader


class DataExplorer(DataLoader):
    def __init__(self, label, csv_dir, index_col=None, normalization_method="minmax", way_of_selection="mrmr"):

        super().__init__(label, csv_dir, index_col=index_col)
        self.normalization_method = normalization_method
        self.x_normalized_df = self.normalize(self.X, method=self.normalization_method)
        self.way_of_selection = way_of_selection

    def plot_preprocess(self,X=None,y=None,
                        features_names=None, 
                        num_features=None
                        ) -> pd.DataFrame:
        """
        This function performs preprocessing on the input data, including normalization and feature selection.
        """
        if X is None and y is None:
            data = self.x_normalized_df
            labels = self.y
        else:
            data = self.normalize(X, method=self.normalization_method)
            labels = y

        if features_names is not None:
                data = data[features_names]

        if num_features is not None:
            selected = self.feature_selection(
                data, labels, method=self.way_of_selection, num_features=num_features
            )
            data = data[selected]
            
        data["labels"] = labels
        return data
    
    def box_illustration(
        self,
        data=None,
        labels=None,
        features_names=None,
        num_features=None,
        dataset_name="dataset"  
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
        
        if self.normalization_method == "minmax":
            data = self.plot_preprocess(
                X=data,
                y=labels,
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
            raise ValueError(f"Unsupported normalization method: {self.normalization_method}. Only 'minmax' is supported for boxplot.")
   
    def pairplots_function(
            self,
            data=None,
            labels=None,
            features_names=None,
            num_features=None
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
                X=data,
                y=labels,
                features_names=features_names,
                num_features=num_features
            )

            sns.pairplot(data, hue="labels")
            plt.show()

    def statistical_difference(
            self,
            data=None,
            labels=None,
            features_names=None,
            num_features=None,
            stat_test="mannwhitneyu",
            p_value=0.05,
            show_box=True
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
            if self.normalization_method == "minmax":
                data = self.plot_preprocess(X=data, y=labels, features_names=features_names, num_features=num_features)
            else:
                raise ValueError(f"Unsupported normalization method: {self.normalization_method}. Only 'minmax' is supported for statistical tests.")

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