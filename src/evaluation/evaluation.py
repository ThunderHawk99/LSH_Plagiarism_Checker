import matplotlib.pyplot as plt
from src.constants import *
from src.common import *
import pandas as pd
import os
import seaborn as sns

## This file contains all the functions to plot/save the results from optimizing the LSH model


# Plot the results of the hyperparameter tuning (we only plot the top 10 values in optimize.py)
def plot_results(results):
    # Create the evaluations directory if it does not exist
    create_dir_if_not_exists(EVALUATION_DIR[0])
    # Create a dataframe from the results
    results_df = pd.DataFrame(
        results,
        columns=[
            "f1",
            "params",
            "precision",
            "recall",
            "false_positives",
            "false_negatives",
            "true_positives",
            "true_negatives",
        ],
    )
    # Plot a barchart to compare the F1 scores of all the hyperparameters
    plot_barchart(results_df)
    # Plot a confusion matrix for each set of hyperparameters
    save_confusion_matrices(results_df)


# Plot a barchart to compare the F1 scores of all the hyperparameters, save it to the evaluations directory under the name 'evaluation.png'
def plot_barchart(results_df):
    # Convert dictionary keys to strings
    results_df["params"] = results_df["params"].apply(lambda x: str(x))
    # Plotting the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(results_df["params"], results_df["f1"])
    plt.xlabel("Params")
    plt.ylabel("F1 Score")
    plt.title("F1 Score by Params")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR[0] + EVALUATION_IMG)
    plt.close()


# Create the confusion matrices for each set of hyperparameters, save them to the evaluations directory under the name 'confusion_matrix_{index}.png'
def save_confusion_matrices(results_df):
    # Create the evaluations directory if it does not exist
    create_dir_if_not_exists(EVALUATION_DIR[0])
    # Iterate through the results and create a confusion matrix for each set of hyperparameters with the given true/false positives/negatives in the results_df
    for index, row in results_df.iterrows():
        params = row["params"]
        false_positives = row["false_positives"]
        false_negatives = row["false_negatives"]
        true_positives = row["true_positives"]  # Access true positives
        true_negatives = row["true_negatives"]  # Access true negatives

        confusion_matrix = [
            [true_positives, false_positives],
            [false_negatives, true_negatives],
        ]

        # Plot confusion matrix
        plt.figure(figsize=(13, 7))  # Adjusted figure size
        sns.heatmap(
            confusion_matrix, annot=True, fmt="d", cmap="Oranges"
        )  # Changed heatmap color to orange
        plt.title(
            f"Confusion Matrix for Params: {params}", fontsize=14
        )  # Adjusted font size of the title
        plt.xlabel("Predicted", fontsize=12)  # Adjusted font size of x-axis label
        plt.ylabel("Actual", fontsize=12)  # Adjusted font size of y-axis label
        plt.xticks(
            ticks=[0.5, 1.5], labels=["Positive", "Negative"], fontsize=10
        )  # Adjusted font size of x-axis ticks and labels
        plt.yticks(
            ticks=[0.5, 1.5], labels=["Positive", "Negative"], fontsize=10
        )  # Adjusted font size of y-axis ticks and labels
        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjusted layout to leave space for the title
        # Save confusion matrix image
        file_name = f"confusion_matrix_{index}.png"
        file_path = os.path.join(EVALUATION_DIR[0], file_name)
        plt.savefig(file_path)
        plt.close()


# Save the results of all hyperparameter combinations, of a specific instance (for example word-based), to a singular CSV file
def save_results_csv(results):
    # Create the evaluations directory if it does not exist
    create_dir_if_not_exists(EVALUATION_DIR[0])
    results_df = pd.DataFrame(
        results,
        columns=[
            "f1",
            "params",
            "precision",
            "recall",
            "false_positives",
            "false_negatives",
            "true_positives",
            "true_negatives",
        ],
    )
    results_df.to_csv(EVALUATION_DIR[0] + EVALUATION_CSV, index=False)
