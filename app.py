from src.LSH.LSHModel import LSHModel
from src.preprocessing.preprocessing import initiate_preprocessing
from src.constants import *
from src.dataset_extraction import corpus_extraction, wikipedia_extraction
from src.optimizing.optimize import optimize
import os
from src.common import *
from src.helpers.helper import optimize_fraud_pair_indexing, index_to_filepath
import time
import csv


# This function is called multiple times to run the pipeline with different configurations
# @return a list containing the time taken for optimization, the time taken for LSH, and the F1-Score
def pipeline(call_idx):
    # We alter the directory for the evaluation files to avoid overwriting previous iterations
    global EVALUATION_DIR
    EVALUATION_DIR[0] = EVALUATION_DIR[0].split("-")[0] + f"-instance-{call_idx}"

    if FRESH_RUN:
        remove_generated_files()

    fraud_pairs = None

    # Get the data from the correct dataset
    if WIKIPEDIA_DATA:
        fraud_pairs = wikipedia_extraction.initiate_dataset_extraction()
    else:
        fraud_pairs = corpus_extraction.initiate_dataset_extraction()

    # Preprocessing the documents
    documents, file_paths = initiate_preprocessing(
        file_paths=[f"{DATASET_DIR}{file}" for file in os.listdir(DATASET_DIR)]
    )

    # Link the fraud pairs to the indexes of the document list
    fraud_pairs_indexed = optimize_fraud_pair_indexing(fraud_pairs, file_paths)

    # Time the duration of hyperparameter tuning
    start_time = time.time()

    ## Hyperparameter tuning
    optimal_hyperparams, best_f1 = optimize(
        documents,
        n_iter=OPTIMISATION_ITER_COUNT,
        fraud_pairs=fraud_pairs_indexed,
        file_paths=file_paths,
    )
    # Calculate the time taken for optimization
    optimize_time = time.time() - start_time

    ## LSH
    optimal_model = LSHModel(**optimal_hyperparams)

    # Time the duration of LSH
    start_lsh_time = time.time()
    # Make the model predict the candidate pairs from the documents
    candidate_pairs = optimal_model.predict(documents, file_paths)

    # Calculate the time taken for LSH
    lsh_time = time.time() - start_lsh_time

    # Print the results
    for pair in candidate_pairs:
        print(f"Similar files: {pair} \n")
    debug_print(f"Fraud pairs: {index_to_filepath(fraud_pairs_indexed, file_paths)}")

    return [optimize_time, lsh_time, best_f1]


def main() -> None:
    #  Access global variables
    global WORD_BASED, KEYWORD_SELECTION_RATIO, OPTIMISATION_ITER_COUNT, FILE_CUTOFF, PARAPHRASE_COUNT

    # Call_idx is used to create new directories to store the results of each instance
    call_idx = 0
    create_dir_if_not_exists(EVALUATION_DIR[0])
    # Write the results for an instance to a CSV file
    with open(COMPLETE_EVALUATION_CSV, "w", newline="", encoding="utf-8-sig") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            [
                f"{OPTIMISATION_ITER_COUNT} iterations, {FILE_CUTOFF} original x {PARAPHRASE_COUNT} paraphrased versions & {FILE_CUTOFF} suspicious files"
            ]
        )
        csv_writer.writerow(
            [
                "Instance",
                "Time taken for optimization (s)",
                "Time taken for LSH (s)",
                "F1-Score (0-1)",
            ]
        )

        # Write instances with altered global variables, to compare the results of different configurations
        # Start with word-based and a keyword ratio of 0.5
        WORD_BASED[0] = True
        KEYWORD_SELECTION_RATIO[0] = 0.5
        csv_writer.writerow(
            [f"Word-based with {KEYWORD_SELECTION_RATIO[0]} keyword ratio"]
            + pipeline(call_idx)
        )
        call_idx += 1

        #  Character-based with a keyword ratio of 0.5
        WORD_BASED[0] = False
        csv_writer.writerow(
            [f"Char-based with {KEYWORD_SELECTION_RATIO[0]} keyword ratio"]
            + pipeline(call_idx)
        )
        call_idx += 1

        #  Character-based with full sentences
        KEYWORD_SELECTION_RATIO[0] = 1
        csv_writer.writerow(["Char-based"] + pipeline(call_idx))
        call_idx += 1

        #  Word-based with full sentences
        WORD_BASED[0] = True
        csv_writer.writerow(["Word-based"] + pipeline(call_idx))


if __name__ == "__main__":
    # Delete the evaluation directory if it exists
    if os.path.exists(EVALUATION_DIR[0]):
        shutil.rmtree(EVALUATION_DIR[0])

    main()
