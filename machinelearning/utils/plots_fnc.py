import pandas as pd
import numpy as np
import os
from itertools import chain
from collections import Counter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from .calc_fnc import _bootstrap_ci

def _plot(
    scores_dataframe: pd.DataFrame, 
    plot: str, 
    scorer: str, 
    final_dataset_name: str
) -> None:
    """
    This function creates a box or violin plot of the outer cross-validation scores for each classifier

    Parameters:
    scores_dataframe (DataFrame): A dataframe containing the results of the outer loop.
    plot (str): The type of plot to create ("box" or "violin").
    scorer (str): The name of the scorer to plot.
    final_dataset_name (str): The name of the dataset.

    Returns:
    None
    """
    scores_long = scores_dataframe.explode(f"{scorer}")
    scores_long[f"{scorer}"] = scores_long[f"{scorer}"].astype(float)
    fig = go.Figure()
    
    classifiers = scores_long["Clf"].unique()

    if plot == "box":
        # Add box plots for each classifier within each Inner_Selection method
        for classifier in classifiers:
            data = scores_long[scores_long["Clf"] == classifier][
                f"{scorer}"
            ]
            median = np.median(data)
            fig.add_trace(
                go.Box(
                    y=data,
                    name=f"{classifier} (Median: {median:.2f})",
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                )
            )

            # Calculate and add 95% CI for the median
            lower, upper = _bootstrap_ci(data, type='median')
            fig.add_trace(
                go.Scatter(
                    x=[f"{classifier} (Median: {median:.2f})",
                    f"{classifier} (Median: {median:.2f})"],
                    y=[lower, upper],
                    mode="lines",
                    line=dict(color="black", dash="dash"),
                    showlegend=False,
                )
            )

    elif plot == "violin":
        for classifier in classifiers:
            data = scores_long[scores_long["Clf"] == classifier][
                f"{scorer}"
            ]
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
            f'The "{plot}" is not a valid option for plotting. Choose between "box" or "violin".'
        )

    # Update layout for better readability
    fig.update_layout(
        autosize = False,
        width=1500,
        height=1200,
        title="Model Selection Results by Classifier",
        yaxis_title=f"Scores {scorer}",
        xaxis_title="Classifier",
        xaxis_tickangle=-45,
        template="plotly_white",
    )
    
    # Save the figure as an image in the "Results" directory
    image_path = f"{final_dataset_name}_model_selection_plot.png"
    fig.write_image(image_path)
            
def _histogram(scores_dataframe, final_dataset_name, freq_feat, clfs, max_features):
    """
    Function to create a histogram of the selected features counts.

    Parameters:
    scores_dataframe (DataFrame): The dataframe containing the results of the outer loop.
    final_dataset_name (str): The name of the dataset.
    freq_feat (int): The number of features to show in the histogram. If None, it will be set to max_features.
    clfs (list): The list of classifiers used.
    max_features (int): The maximum number of features.

    Returns:
    None
    """
    if freq_feat is None:
        freq_feat = max_features
    elif freq_feat > max_features:
        freq_feat = max_features

    # Plot histogram of features
    feature_counts = Counter()
    for idx, row in scores_dataframe.iterrows():
        if row["Sel_way"] != "none":  # If no features were selected, skip
            features = list(
                chain.from_iterable(
                    [list(index_obj) for index_obj in row["Sel_feat"]]
                )
            )
            feature_counts.update(features)

    sorted_features_counts = feature_counts.most_common()

    if len(sorted_features_counts) == 0:
        print("No features were selected.")
    else:
        features, counts = zip(*sorted_features_counts[:freq_feat])
        counts = [x / len(clfs) for x in counts]  # Normalize counts
        print(f"Selected {freq_feat} features")

        # Create the bar chart using Plotly
        fig = go.Figure()

        # Add bars to the figure
        fig.add_trace(go.Bar(
            x=features,
            y=counts,
            marker=dict(color="skyblue"),
            text=[f"{count:.2f}" for count in counts],  # Show normalized counts as text
            textposition='auto'
        ))

        # Set axis labels and title
        fig.update_layout(
            title="Histogram of Selected Features",
            xaxis_title="Features",
            yaxis_title="Counts",
            xaxis_tickangle=-90,  # Rotate x-ticks to avoid overlap
            bargap=0.2,
            template="plotly_white",
            width=min(max(1000, freq_feat * 20), 2000),  # Dynamically adjust plot width
            height=700  # Set plot height
        )

        # Save the plot to 'Results/histogram.png'
        save_path = f"{final_dataset_name}_histogram.png"
        fig.write_image(save_path)

def _eval_boxplot(self, estimator_name, eval_df, splits, evaluation):
    """
    Generate a boxplot to visualize the model evaluation results.
    
    :param estimator_name: The name of the estimator.
    :type estimator_name: str
    :param eval_df: The evaluation dataframe containing the scores.
    :type eval_df: pandas.DataFrame
    :param cv: The number of cross-validation folds.
    :type cv: int
    :param evaluation: The evaluation method to use ("bootstrap", "cv_simple", or any other custom method).
    :type evaluation: str
    """
    
    results_dir = "Final_Model_Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if (evaluation == "bootstrap") or (evaluation == "oob") or (evaluation == "train_test"):
        fig = go.Figure()
        fig.add_trace(
            go.Box(
                y=eval_df["Scores"],
                name=estimator_name,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                boxmean=True,
            )
        )

        fig.update_layout(
            title=f"Model Evaluation Results With {evaluation} Method",
            yaxis_title="Score",
            template="plotly_white",
        )

    elif evaluation == "cv_rounds":
        # fig = make_subplots(
        #     rows=1, cols=2, subplot_titles=("Summary Boxplot", "Rounds Scores")
        # )
        fig = go.Figure()
        all_scores = []
        best_scores_rounds = []
        best_cv = []
        for row in range(eval_df.shape[0]):
            for score in eval_df["Scores"].iloc[row]:
                all_scores.append(score)

        fig.add_trace(
            go.Box(
                y=all_scores,
                name="All trial Scores",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            )#,
            # row=1,
            # col=1,
        )
            
        # Update layout for better readability
        fig.update_layout(
            title=f"Model Evaluation Results With {evaluation} Method",
            height=600,
            width=1200,
            showlegend=False,
        )

    elif evaluation == "train_test":
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=eval_df["round"],  # You can change this to any x-axis variable you want
                y=eval_df["Scores"],     # Y-axis variable
                mode='markers',          # Use markers for scatter plot
                name=estimator_name,
            )
        )

        fig.update_layout(
            title=f"Model Evaluation Results With {evaluation} Method",
            xaxis_title="Score type",  # Change this to the relevant label for the x-axis
            yaxis_title="Score",
            template="plotly_white",
        )
    
    fig.show()

    # Save the plot to 'Results/final_model_evaluation.png'
    save_path = os.path.join(results_dir, f"final_model_evaluation_for_{estimator_name}.png")
    fig.write_image(save_path)