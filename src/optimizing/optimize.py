from sklearn.model_selection import ParameterSampler
from src.LSH.LSHModel import LSHModel
from src.constants import *
from multiprocessing import Pool
from src.evaluation.evaluation import plot_results, save_results_csv
import numpy as np
import gc
from src.helpers.helper import timeit
from tqdm import tqdm
from src.constants import RANDOM_SEED

# Define the hyperparameter grid for the LSH model to be used in hyperparameter tuning
param_grid = {
    "shingle_size": SHINGLE_SIZE,
    "window_step": WINDOW_STEP,
    "n_hash": N_HASH,
    "n_bands": N_BANDS,
    "K": K,
}


# Perform LSH with a certain hyperparameter configuration
# @return the F1 score, precision, recall, false positives, false negatives, true positives, true negatives of the model
def score_with_params(args):
    # Unpack the arguments
    docs, params, ground_truth, file_paths = args

    # Create the LSH model with the specified hyperparameters and score it
    model = LSHModel(**params)
    (
        f1,
        precision,
        recall,
        false_positives,
        false_negatives,
        true_positives,
        true_negatives,
    ) = model.score(docs, ground_truth, file_paths=file_paths)

    # Clear all but f1, params
    del model
    del ground_truth
    del docs
    del file_paths
    gc.collect()

    return (
        f1,
        params,
        precision,
        recall,
        false_positives,
        false_negatives,
        true_positives,
        true_negatives,
    )


# Perform hyperparameter tuning to find the best hyperparameters
# @return the best hyperparameters and the corresponding F1 score
@timeit
def optimize(docs, n_iter=3, fraud_pairs=None, file_paths=None):
    print("Starting optimisation process...")
    best_f1 = 0  # Store the best F1 score found so far
    best_params = None  # Corresponding best hyperparameters of best f1 score
    f1_scores = []  # Store F1 scores
    param_combinations = []  # Store parameter combinations
    results = []  # Store results

    # Use numpy's random number generator to ensure reproducibility
    # Create the random number generator with the specified seed
    rng = np.random.default_rng(RANDOM_SEED)
    random_state = rng.integers(0, 10000)

    # Sample hyperparameters randomly without replacement
    param_sampler = list(
        ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state)
    )

    del rng
    del random_state

    # Prepare the arguments for parallel execution
    args_list = [
        (docs, params, fraud_pairs, file_paths) for params in param_sampler
    ]  # Include fraud_pairs & file paths in each tuple
    with Pool(processes=AMOUNT_OF_WORKERS) as pool:
        # Process results as soon as they are available
        for (
            f1,
            params,
            precision,
            recall,
            false_positives,
            false_negatives,
            true_positives,
            true_negatives,
        ) in tqdm(
            pool.imap_unordered(score_with_params, args_list), total=len(args_list)
        ):  # Iterate over the results and update the best F1 score if a better one is found
            if f1 > best_f1:
                best_f1 = f1
                best_params = params
                tqdm.write(f"New Best F1: {f1} with Params: {params}")

            # Continue to append results to keep track of all iterations
            results.append(
                (
                    f1,
                    params,
                    precision,
                    recall,
                    false_positives,
                    false_negatives,
                    true_positives,
                    true_negatives,
                )
            )
            f1_scores.append(f1)
            param_combinations.append(params)

    if not best_params:
        print("No optimal hyperparameters found")
        return

    # Sort the results by F1 score in descending order
    sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
    top_results = sorted_results[:TOP_VALUES]

    # Save the results to a CSV file and plot the top {TOP_VALUES} results
    save_results_csv(results)
    plot_results(top_results)
    return best_params, best_f1
