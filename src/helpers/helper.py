import time
import functools
from src.common import debug_print
from src.constants import PARAPHRASED, WIKIPEDIA_DATA
import pandas as pd


# Iterate over the candidate pairs and filter out pairs with the same document number
# @return the filtered pairs
def filter_pairs_by_number(candidate_pairs, file_paths):
    # Depending on which dataset we are using, the file names (and therefore information) are structured differently
    # Extract the file number and type from the file path
    def get_file_details(file_idx):
        parts = file_paths[file_idx].split("/")
        number, file_type = (
            parts[2].split(".txt")[0].split("-")
            if WIKIPEDIA_DATA
            else (
                parts[2].split(".txt")[0][-5:],
                parts[2].split(".txt")[0].split("-")[0],
            )
        )
        return int(number), file_type

    # Initialize a set to store the filtered pairs
    filtered_pairs = set()

    for pair in candidate_pairs:
        # Get the file number and type for each file in the pair
        file1_number, file1_type = get_file_details(pair[0])
        file2_number, file2_type = get_file_details(pair[1])

        # If the file numbers or types are different, add the pair to the filtered pairs.
        # This means we do not allow files to be compared with themselves or (paraphrased) versions of themselves
        if file1_number != file2_number or file1_type != file2_type:
            # It is possible to compare a paraphrased file with the original file
            # We then want to return the index of the original file rather than the paraphrased file for the candidate pairs
            original_pair = original_idx(pair[0], pair[1], file_paths)
            filtered_pairs.add(original_pair)

    return filtered_pairs


# Get the original index of the files if they are paraphrased
# @return the index of the original file from which the file is paraphrased (if one of the files is paraphrased)
def original_idx(file1, file2, file_paths):
    # Get the original index of the files if they are paraphrased
    def get_original_index(file_idx):
        file_path = file_paths[file_idx]
        if PARAPHRASED in file_path:
            # We always name the files: original_file_paraphrased_version.txt
            # Paraphrased filepaths are structured as: assets/paraphrased/original_file/original_file_paraphrased_version.txt
            # Therefore the last digit reflects the paraphrased version
            paraphrase_version = int(file_path.split("/")[3].split(".txt")[0][-1])
            # The index of the original file is the index of the paraphrased file minus the paraphrased version + 1 as we start counting from 0
            return file_idx - (paraphrase_version + 1)
        return file_idx

    # Get the original index for both files
    original_file1_idx = get_original_index(file1)
    original_file2_idx = get_original_index(file2)

    # Make sure the smallest index is first
    min_idx = min(original_file1_idx, original_file2_idx)
    max_idx = max(original_file1_idx, original_file2_idx)
    return (min_idx, max_idx)


# Decorator to time the duration of a function and display the time taken
# @return: nothing, print statement.
def timeit(method):
    @functools.wraps(method)
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        debug_print(f"{method.__name__} : {(te - ts)} sec")
        return result

    return timed


# Read the fraud pairs and return a new set containing the same fraud pairs, but with their corresponding indexes of the documents
# which can be found in file_paths.
# @return the fraud pairs with their corresponding indexes
def optimize_fraud_pair_indexing(fraud_pairs, file_paths):
    # Now link the fraud pairs to the indexes of the document list
    fraud_pairs_index = set()
    for fraud_pair in fraud_pairs:
        # Without the .txt extension
        original_file = fraud_pair[0].replace(".txt", "")
        fraud_file = fraud_pair[1].replace(".txt", "")
        # Find in file_paths all occurrences of the original and fraud file
        original_file_paths = set()
        fraud_file_paths = set()

        # In-case each file is paraphrased or not, we would need all occurrences of the original and fraud file
        for file_path in file_paths:
            if original_file in file_path:
                original_file_paths.add(file_path)
            if fraud_file in file_path:
                fraud_file_paths.add(file_path)

        # Now combine the original and fraud file paths
        for original_file_path in original_file_paths:
            # Filter out the paraphrased files
            fraud_file_paths = [
                file for file in fraud_file_paths if PARAPHRASED not in file
            ]
            for fraud_file_path in fraud_file_paths:
                og_file_idx = file_paths.index(original_file_path)
                fr_file_idx = file_paths.index(fraud_file_path)

                min_idx = min(og_file_idx, fr_file_idx)
                max_idx = max(og_file_idx, fr_file_idx)
                fraud_pairs_index.add(
                    (
                        min_idx,
                        max_idx,
                    )
                )
    return fraud_pairs_index


# Convert an index to the corresponding file path for that document using the file_paths list
# @return the file paths corresponding to the indices
def index_to_filepath(candidate_pairs, file_paths):
    # Convert the indices to file paths
    candidate_pairs = set(
        [(file_paths[pair[0]], file_paths[pair[1]]) for pair in candidate_pairs]
    )
    return candidate_pairs
