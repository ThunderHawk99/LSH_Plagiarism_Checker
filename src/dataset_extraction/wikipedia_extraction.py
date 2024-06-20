from src.constants import *
import os
import shutil
from src.common import create_dir_if_not_exists
import xml.etree.ElementTree as ET
from collections import defaultdict
from src.constants import FILE_CUTOFF


def extract_files_from_corpus():
    original_idx = 0
    suspect_idx = 0

    # All textfiles we want are in "external-detection-corpus"
    folder = os.path.join(WIKIPEDIA_DIR)

    # See if the folder can be found
    if not os.path.exists(folder):
        return

    # Index to document name mapping
    # e.g. 1 -> "source-document00001.txt"
    index_to_name_pairs = {}
    name_to_index_pairs = {}
    # fraud pairs mapping
    # [] -> [(1, 2), (2, 3), (3, 4)]
    fraud_pairs = set()

    # Not added yet queue
    queue = []

    # Now loop over all .txt files that can be found and put them
    # in the ASSETS_DIR/DATASET_DIR/ folder
    for dir_path, _, files in os.walk(folder):
        for file in files:
            # Filenames formatted as "339-ORIG.txt" or "339-SPUN.txt"
            # Each file starts with a arbitrarily long digit number, so remove the -ORIG.txt or -SPUN.txt to extract the number
            digit = file.split("-")[0]

            # Only keep a certain number of files
            is_original = "-ORIG.txt" in file
            check_on_digit = digit.isdigit()

            if is_original:
                check_on_digit = check_on_digit and original_idx < FILE_CUTOFF
            else:
                check_on_digit = check_on_digit and suspect_idx < FILE_CUTOFF

            if check_on_digit or EXTRACT_ALL_FILES:
                file_path = os.path.join(dir_path, file)
                dest_path = os.path.join(DATASET_DIR, file)

                # Only copy the file if it does not exist
                if not os.path.exists(dest_path):
                    shutil.copy(file_path, dest_path)

                # Add both indexes together to create a unique index
                combined_index = original_idx + suspect_idx
                index_to_name_pairs[combined_index] = file
                name_to_index_pairs[file] = combined_index

                # If it has -SPUN in the name, start adding it to the fraud pairs
                if "-SPUN.txt" in file:
                    suspect_idx += 1
                    # Check if its -ORIG.txt is already in the dataset
                    orig_file = file.replace("-SPUN.txt", "-ORIG.txt")
                    if orig_file in name_to_index_pairs:
                        fraud_pairs.add((orig_file, file))
                    else:
                        queue.append(file)
                else:
                    original_idx += 1

    # Write the fraud pairs to a file
    with open(FRAUD_PAIRS_FILE, "w", encoding="utf-8-sig") as f:
        for pair in fraud_pairs:
            f.write(f"{pair[0]} {pair[1]}\n")

    return fraud_pairs


def setup():
    # Create the directories if they do not exist
    create_dir_if_not_exists(DATASET_DIR)
    create_dir_if_not_exists(PROCESSED_DIR)


def initiate_dataset_extraction():
    setup()
    return extract_files_from_corpus()
