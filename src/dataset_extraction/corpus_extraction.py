from src.constants import *
import os
import shutil
from src.common import create_dir_if_not_exists
import xml.etree.ElementTree as ET
from collections import defaultdict
from src.constants import FILE_CUTOFF


def extract_files_from_corpus():
    index = 0

    # All textfiles we want are in "external-detection-corpus"
    folder = os.path.join(CORPUS_DIR, "external-detection-corpus")

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
        # Skip the file names "retrieval-task"
        files = [file for file in files if "retrieval-task" not in file]
        for file in files:
            # Filenames formatted as "source-document05501.txt" or "suspicious-document05501.txt"
            # Each file ends with a 5 digit number e.g. "00001"
            # Only keep a certain number of files
            digit = file[-9:-4]
            check_on_digit = digit.isdigit() and int(digit) <= FILE_CUTOFF

            if check_on_digit or EXTRACT_ALL_FILES:
                file_path = os.path.join(dir_path, file)
                is_text_file = file.endswith(".txt")
                is_xml_file = file.endswith(".xml")
                dest_dir = DATASET_DIR if is_text_file else DATASET_METADATA_DIR
                dest_path = os.path.join(dest_dir, file)

                if is_text_file:
                    # Only copy the file if it does not exist
                    if not os.path.exists(dest_path):
                        shutil.copy(file_path, dest_path)

                    index_to_name_pairs[index] = file
                    name_to_index_pairs[file] = index
                    index += 1
                elif is_xml_file:
                    # Only copy the file if it does not exist
                    if not os.path.exists(dest_path):
                        shutil.copy(file_path, dest_path)
                    queue.append(file_path)

    # Either read the fraud pairs from the file or process the XML files to get them
    fraud_pairs = process_fraud_pairs(queue, name_to_index_pairs, index_to_name_pairs)

    # Write the fraud pairs to a file
    with open(FRAUD_PAIRS_FILE, "w", encoding="utf-8-sig") as f:
        for pair in fraud_pairs:
            f.write(f"{pair[0]} {pair[1]}\n")

    return list(fraud_pairs)


# This function is used to process the fraud pairs to return a list of tuples containing the fraud pairs
def process_fraud_pairs(queue, name_to_index_pairs, index_to_name_pairs):
    fraud_pairs = set()
    # Check if the FRAUD_PAIRS_FILE exists
    if os.path.exists(FRAUD_PAIRS_FILE):
        # If the file exists, read the data and populate fraud_pairs
        with open(FRAUD_PAIRS_FILE, "r", encoding="utf-8-sig") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    fraud_pairs.add((parts[0], parts[1]))
    else:
        # If the file does not exist, perform the loop to parse XML files and gather fraud pairs
        for file_path in queue:
            try:
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    tree = ET.parse(f)
                    root = tree.getroot()

                    # Get the document's reference attribute
                    fraud_doc = root.attrib.get("reference")
                    curr_fraud_doc_index = name_to_index_pairs.get(fraud_doc, -1)

                    if curr_fraud_doc_index == -1:
                        continue

                    # Go through all features in the XML file
                    for feature in root.findall("feature"):
                        # Check if the feature is plagiarism
                        if feature.attrib.get("name") == "plagiarism":
                            # Get the source reference attribute
                            source_reference = feature.attrib.get("source_reference")
                            # Get the source index. IF this is not found, this means that it did not make it up to the cutoff
                            source_index = name_to_index_pairs.get(source_reference, -1)

                            if source_index == -1:
                                continue

                            lower_idx = min(curr_fraud_doc_index, source_index)
                            higher_idx = max(curr_fraud_doc_index, source_index)
                            fraud_pairs.add(
                                (
                                    index_to_name_pairs[lower_idx],
                                    index_to_name_pairs[higher_idx],
                                )
                            )

            except Exception as e:
                print("Error processing file:", file_path, "Error:", e)

        # After collecting all new fraud pairs, remove duplicates and write to file
        fraud_pairs = list(fraud_pairs)
        with open(FRAUD_PAIRS_FILE, "w", encoding="utf-8-sig") as f:
            for pair in fraud_pairs:
                f.write(f"{pair[0]} {pair[1]}\n")

    return fraud_pairs


def setup():
    # Create the directories if they do not exist
    create_dir_if_not_exists(DATASET_DIR)
    create_dir_if_not_exists(DATASET_METADATA_DIR)
    create_dir_if_not_exists(PROCESSED_DIR)


def initiate_dataset_extraction():
    setup()
    return extract_files_from_corpus()
