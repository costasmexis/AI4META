from typing import Optional, List, Dict, Union
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from itertools import chain
from collections import Counter
from src.utils.statistics.bootstrap_ci import _calc_ci_btstrp

def _plot_per_clf(
    scores_dataframe: pd.DataFrame,
    plot: str,
    scorer: str,
    final_dataset_name: str
) -> None:
    """
    Create visualization of classifier performance scores.

    Generates either box or violin plots showing the distribution of scores
    for each classifier, including confidence intervals for box plots.

    Parameters
    ----------
    scores_dataframe : pd.DataFrame
        DataFrame containing model evaluation scores
    plot : str
        Plot type ('box' or 'violin')
    scorer : str
        Name of the scoring metric to plot
    final_dataset_name : str
        Base name for saving the plot file

    Raises
    ------
    ValueError
        If invalid plot type is specified

    Notes
    -----
    - Box plots include 95% confidence intervals for medians
    - Both plot types show individual data points
    - Plots are automatically saved to the results/images directory
    """
    # Prepare data
    scores_long = scores_dataframe.explode(scorer)
    scores_long[scorer] = scores_long[scorer].astype(float)
    
    # Initialize plot
    fig = go.Figure()
    classifiers = scores_long["Clf"].unique()

    if plot == "box":
        # Create box plots with confidence intervals
        for classifier in classifiers:
            # Get data for current classifier
            data = scores_long[scores_long["Clf"] == classifier][scorer]
            median = np.median(data)

            # Add box plot
            fig.add_trace(
                go.Box(
                    y=data,
                    name=f"{classifier} (Median: {median:.2f})",
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                )
            )

            # Add confidence interval
            lower_ci, upper_ci = _calc_ci_btstrp(data, central_tendency='median')
            fig.add_trace(
                go.Scatter(
                    x=[f"{classifier} (Median: {median:.2f})"]*2,
                    y=[lower_ci, upper_ci],
                    mode="lines",
                    line=dict(color="black", dash="dash"),
                    showlegend=False,
                )
            )

    elif plot == "violin":
        # Create violin plots
        for classifier in classifiers:
            data = scores_long[scores_long["Clf"] == classifier][scorer]
            median = np.median(data)
            fig.add_trace(
                go.Violin(
                    y=data,
                    name=f"{classifier} (Median: {median:.2f})",
                    box_visible=False,
                    points="all",
                    jitter=0.3,
                    pointpos=-1.8,
                )
            )
    else:
        raise ValueError(
            f"Invalid plot type '{plot}'. Use 'box' or 'violin'."
        )

    # Configure layout
    fig.update_layout(
        autosize=False,
        width=2500,
        height=1900,
        title="Model Selection Results by Classifier",
        yaxis_title=f"Scores {scorer}",
        xaxis_title="Classifier",
        xaxis_tickangle=-45,
        template="plotly_white",
    )

    # Save plot
    image_path = f"{final_dataset_name}_model_selection_plot.png"
    fig.write_image(image_path)

def _plot_per_metric(
    scores_df: pd.DataFrame,
    name: str
) -> None:
    """
    Generate boxplot visualization for multiple evaluation metrics.

    Parameters
    ----------
    scores_df : pd.DataFrame
        DataFrame containing evaluation metrics
    name : str
        Base name for the plot and saved file

    Notes
    -----
    Generates an interactive plot showing distribution of all metrics
    and saves it as PNG
    """
    fig = go.Figure()

    # Add box plot for each metric
    for metric in scores_df.columns:
        fig.add_trace(go.Box(
            y=scores_df[metric],
            name=metric
        ))

    # Configure layout
    fig.update_layout(
        autosize=False,
        width=1500,
        height=1200,
        title=name,
        yaxis_title="Scores",
        xaxis_title="Metrics",
        xaxis_tickangle=-45,
        template="plotly_white"
    )

    # Display and save plot
    fig.show()
    fig.write_image(f"{name}.png")

def _histogram(
    scores_dataframe: pd.DataFrame,
    final_dataset_name: str,
    freq_feat: Optional[int],
    clfs: List[str],
    max_features: int
) -> None:
    """
    Create histogram visualization of feature selection frequencies.

    Parameters
    ----------
    scores_dataframe : pd.DataFrame
        DataFrame containing feature selection results
    final_dataset_name : str
        Base name for saving the plot
    freq_feat : int, optional
        Number of top features to display
    clfs : list[str]
        List of classifiers used
    max_features : int
        Maximum number of features available

    Notes
    -----
    - Normalizes feature counts by number of classifiers
    - Only includes features selected through feature selection methods
    - Automatically adjusts plot width based on number of features
    """
    # Handle freq_feat parameter
    if freq_feat is None or freq_feat > max_features:
        freq_feat = max_features

    # Count feature occurrences
    feature_counts = Counter()
    for _, row in scores_dataframe.iterrows():
        if row["Sel_way"] != "none":
            features = list(chain.from_iterable(
                [list(index_obj) for index_obj in row["Sel_feat"]]
            ))
            feature_counts.update(features)

    # Check if any features were selected
    if not feature_counts:
        print("No features were selected.")
        return

    # Process feature counts
    top_features = feature_counts.most_common(freq_feat)
    features, counts = zip(*top_features)
    normalized_counts = [count / len(clfs) for count in counts]
    print(f"Selected {freq_feat} features")

    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=features,
        y=normalized_counts,
        marker=dict(color="skyblue"),
        text=[f"{count:.2f}" for count in normalized_counts],
        textposition='auto'
    ))

    # Configure layout
    fig.update_layout(
        title="Histogram of Selected Features",
        xaxis_title="Features",
        yaxis_title="Normalized Counts",
        xaxis_tickangle=-90,
        bargap=0.2,
        template="plotly_white",
        width=min(max(1000, freq_feat * 20), 2000),
        height=700
    )

    # Save plot
    save_path = f"{final_dataset_name}_histogram.png"
    fig.write_image(save_path)

# import pandas as pd
# import numpy as np
# import os
# from itertools import chain
# from collections import Counter
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from src.utils.statistics.bootstrap_ci import _calc_ci_btstrp

# def _plot_per_clf(
#     scores_dataframe: pd.DataFrame, 
#     plot: str, 
#     scorer: str, 
#     final_dataset_name: str
# ) -> None:
#     """
#     This function creates a box or violin plot of the outer cross-validation scores for each classifier

#     Parameters:
#     scores_dataframe (DataFrame): A dataframe containing the results of the outer loop.
#     plot (str): The type of plot to create ("box" or "violin").
#     scorer (str): The name of the scorer to plot.
#     final_dataset_name (str): The name of the dataset.

#     Returns:
#     None
#     """
#     results_dir = "results/images"
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
        
#     scores_long = scores_dataframe.explode(f"{scorer}")
#     scores_long[f"{scorer}"] = scores_long[f"{scorer}"].astype(float)
#     fig = go.Figure()
    
#     classifiers = scores_long["Clf"].unique()

#     if plot == "box":
#         # Add box plots for each classifier within each Inner_Selection method
#         for classifier in classifiers:
#             data = scores_long[scores_long["Clf"] == classifier][
#                 f"{scorer}"
#             ]
#             median = np.median(data)
#             fig.add_trace(
#                 go.Box(
#                     y=data,
#                     name=f"{classifier} (Median: {median:.2f})",
#                     boxpoints="all",
#                     jitter=0.3,
#                     pointpos=-1.8,
#                 )
#             )

#             # Calculate and add 95% CI for the median
#             lower, upper = _calc_ci_btstrp(data, central_tendency='median')
#             fig.add_trace(
#                 go.Scatter(
#                     x=[f"{classifier} (Median: {median:.2f})",
#                     f"{classifier} (Median: {median:.2f})"],
#                     y=[lower, upper],
#                     mode="lines",
#                     line=dict(color="black", dash="dash"),
#                     showlegend=False,
#                 )
#             )

#     elif plot == "violin":
#         for classifier in classifiers:
#             data = scores_long[scores_long["Clf"] == classifier][
#                 f"{scorer}"
#             ]
#             median = np.median(data)
#             fig.add_trace(
#                 go.Violin(
#                     y=data,
#                     name=f"{classifier} (Median: {median:.2f})",
#                     box_visible=False,
#                     points="all",
#                     jitter=0.3,
#                     pointpos=-1.8,
#                 )
#             )
#     else:
#         raise ValueError(
#             f'The "{plot}" is not a valid option for plotting. Choose between "box" or "violin".'
#         )

#     # Update layout for better readability
#     fig.update_layout(
#         autosize = False,
#         width=1500,
#         height=1200,
#         title="Model Selection Results by Classifier",
#         yaxis_title=f"Scores {scorer}",
#         xaxis_title="Classifier",
#         xaxis_tickangle=-45,
#         template="plotly_white",
#     )
    
#     # Save the figure as an image in the "Results" directory
#     image_path = f"{final_dataset_name}_model_selection_plot.png"
#     fig.write_image(image_path)
            
# def _plot_per_metric(scores_df, name):
#     """
#     Generate a boxplot to visualize the model evaluation results.
    
#     Parameters:
#     scores_df (pandas.DataFrame): A DataFrame containing the model evaluation results.
#     name (str): The name of the dataset.
    
#     Returns:
#     None
#     """

#     fig = go.Figure()
    
#     # Add a boxplot for each metric
#     for metric in scores_df.columns:
#         fig.add_trace(go.Box(y=scores_df[metric], name=metric))

#     # Update layout for better readability
#     fig.update_layout(
#         autosize = False,
#         width=1500,
#         height=1200,
#         title=f"{name}",
#         yaxis_title=f"Scores",
#         xaxis_title="Metrics",
#         xaxis_tickangle=-45,
#         template="plotly_white"
#     )
#     fig.show()

#     # Save the plot to 'Results/final_model_evaluation.png'
#     # save_path = os.path.join(results_dir, f"evaluation_{estimator_name}_{evaluation}_{inner_selection}_{dataset_plot_name}.png")
#     fig.write_image(f"{name}.png")

# def _histogram(scores_dataframe, final_dataset_name, freq_feat, clfs, max_features):
#     """
#     Function to create a histogram of the selected features counts.

#     Parameters:
#     scores_dataframe (DataFrame): The dataframe containing the results of the outer loop.
#     final_dataset_name (str): The name of the dataset.
#     freq_feat (int): The number of features to show in the histogram. If None, it will be set to max_features.
#     clfs (list): The list of classifiers used.
#     max_features (int): The maximum number of features.

#     Returns:
#     None
#     """
#     if freq_feat is None:
#         freq_feat = max_features
#     elif freq_feat > max_features:
#         freq_feat = max_features

#     # Plot histogram of features
#     feature_counts = Counter()
#     for idx, row in scores_dataframe.iterrows():
#         if row["Sel_way"] != "none":  # If no features were selected, skip
#             features = list(
#                 chain.from_iterable(
#                     [list(index_obj) for index_obj in row["Sel_feat"]]
#                 )
#             )
#             feature_counts.update(features)

#     sorted_features_counts = feature_counts.most_common()

#     if len(sorted_features_counts) == 0:
#         print("No features were selected.")
#     else:
#         features, counts = zip(*sorted_features_counts[:freq_feat])
#         counts = [x / len(clfs) for x in counts]  # Normalize counts
#         print(f"Selected {freq_feat} features")

#         # Create the bar chart using Plotly
#         fig = go.Figure()

#         # Add bars to the figure
#         fig.add_trace(go.Bar(
#             x=features,
#             y=counts,
#             marker=dict(color="skyblue"),
#             text=[f"{count:.2f}" for count in counts],  # Show normalized counts as text
#             textposition='auto'
#         ))

#         # Set axis labels and title
#         fig.update_layout(
#             title="Histogram of Selected Features",
#             xaxis_title="Features",
#             yaxis_title="Counts",
#             xaxis_tickangle=-90,  # Rotate x-ticks to avoid overlap
#             bargap=0.2,
#             template="plotly_white",
#             width=min(max(1000, freq_feat * 20), 2000),  # Dynamically adjust plot width
#             height=700  # Set plot height
#         )

#         # Save the plot to 'results/histogram.png'
#         save_path = f"{final_dataset_name}_histogram.png"
#         fig.write_image(save_path)