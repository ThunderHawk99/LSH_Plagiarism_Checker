import os
import shutil
from src.constants import *
from tqdm import tqdm


# Create a directory if it does not exist
# @return True if the directory was created, False otherwise
def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return True

    return False


# Read data from a file
# @return the data that was read from the file, or None if an error occurred. Also return the file name e.g. "file.txt"
def read_in_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8-sig") as file:
            # Also return the file name, also return the file extension (will be empty string if there is none)
            return file.read(), os.path.basename(file_path)
    except FileNotFoundError:
        # This should NOT happen, as we are looping through the files in the directory
        print(f"(CRITICAL ERROR!): The file {file_path} does not exist.")
        return None
    except Exception as e:
        # Something else can go wrong, wrong encoding, etc.
        print(
            f"(INVESTIGATE!) An error occurred while reading the file {file_path}: {e}"
        )
        return None, None


# Write data to a file
# @return the data that was written to the file, or None if an error occurred
def write_to_file(file_path, data):
    try:
        with open(file_path, "w", encoding="utf-8-sig") as file:
            file.write(data)
        # print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to save data to {file_path}: {e}")
        return None

    return data


def get_file_extension(full_file_name):
    return os.path.splitext(full_file_name)[1]


# Custom print function for debugging such that based on {DEBUG} variable, it will print the message.
# Use tqdm.write instead of print to avoid issues with tqdm progress bar.
def debug_print(message):
    if DEBUG:
        tqdm.write(message)


def get_paraphrased_paths():
    paraphrased_file_paths = []

    # Iterate over each item in PARAPHRASED_DIR
    for item in os.listdir(PARAPHRASED_DIR):
        item_path = os.path.join(PARAPHRASED_DIR, item)
        if os.path.isdir(item_path):
            # Iterate over each file in the subdirectory
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)
                paraphrased_file_paths.append(file_path)

    return paraphrased_file_paths


def get_file_paths():
    preprocessed_file_paths = [
        os.path.join(PREPROCESSED_DIR, file) for file in os.listdir(PREPROCESSED_DIR)
    ]
    paraphrased_file_paths = get_paraphrased_paths()
    file_paths = preprocessed_file_paths + paraphrased_file_paths

    return file_paths


def read_from_file_paths(file_paths):
    documents = []
    for file_path in file_paths:
        data, _ = read_in_file(file_path)
        documents.append(data)

    return documents


# Clean up the generated files and directories
def remove_generated_files():
    print("Removing generated files...")

    # Remove all these:
    directories_to_remove = [
        DATASET_DIR,
        PREPROCESSED_DIR,
        PROCESSED_DIR,
        PARAPHRASED_DIR,
        # EVALUATION_DIR[0],
        DATASET_METADATA_DIR,  # This is needed in-case the Corpus dataset is used
    ]

    # Delete the whole directory
    for directory in directories_to_remove:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"Deleted {directory}")
            except Exception as e:
                print(f"Error deleting {directory}: {e}")
