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
        width=7000,
        height=2500,
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
