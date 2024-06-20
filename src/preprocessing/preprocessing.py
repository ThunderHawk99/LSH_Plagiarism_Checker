from multiprocessing import Pool
from src.constants import *
from src.common import *
import re
from src.preprocessing.preprocess_text import process_single_text_file
from src.preprocessing.preprocess_code import process_single_code_file


def determine_file_type(data, full_file_name):
    file_extension = get_file_extension(full_file_name)
    if file_extension:
        # If File extension is known, we can determine the type of data based on the extension also return the language
        if file_extension in CODE_EXTENSIONS:
            return "code:" + CODE_EXTENSIONS[file_extension]
        elif file_extension in TEXT_EXTENSIONS:
            return "text"
    else:
        # We can create a simple pattern recognition system to determine the type of data.
        # Many python files will start with 'import' or 'from', while text files will not.
        # These will also contain lots of 'def', 'class', and 'return' keywords.

        # Java files will contain 'public', 'private', 'class', 'void', and 'return' keywords.

        # However, it is possible for a text file to contain these keywords, so we will need to
        # be careful with our pattern recognition and first check if the file is a code file.

        # Create a dictionary of programming patterns using regular expressions
        # Note: \s is a space character, \b is a word boundary, and \w is a word character
        programming_patterns = {
            "python": [
                r"\s*import\s+\b",  # import
                r"\s*from\s+\w+\s+import\b",  # from ... import
                r"\s*def\s+\w+\s*\(b",  # def (
                r"\s*print\(",  # print(
            ],
            "java": [
                r"\bpublic\s+class\b",  # public class
                r"\bpublic\s+abstract class\b",  # public abstract class
                r"\bprivate\s+\b",  # private
                r"\bvoid\s+\b",  # void
                r"\bout.println\b",  # out.println
            ],
        }

        for language in programming_patterns:
            # Loop through each pattern
            for pattern in programming_patterns[language]:
                # Check if the pattern is in the data
                if re.search(pattern, data):
                    return "code:" + language

        return "text"


# Process single file
def process_single_file(file_path):
    # Read in the file
    data, full_file_name = read_in_file(file_path)

    # See what type of data it is
    file_type = determine_file_type(data, full_file_name)

    # Split the file type
    data_split = file_type.split(":")
    if data_split[0] == "text":
        return process_single_text_file(file_path, data, full_file_name)

    elif data_split[0] == "code":
        language = data_split[1]
        return process_single_code_file(file_path, language, data, full_file_name)
    else:
        # If the file type is unknown, we will not process it
        print(f"Unknown file type: {file_path}")

    return data


# Start the preprocessing
def start_preprocessing(file_paths):

    with Pool(processes=AMOUNT_OF_WORKERS) as pool:
        nested_results = pool.map(process_single_file, file_paths)

    # Flatten the list of lists into a single list of tuples
    results = [item for sublist in nested_results for item in sublist]

    # Separate results into documents and file paths
    documents, file_paths = zip(*results) if results else ([], [])

    return list(documents), list(file_paths)


def setup():
    # Create the directories if they do not exist
    create_dir_if_not_exists(PREPROCESSED_DIR)
    create_dir_if_not_exists(PARAPHRASED_DIR)


def initiate_preprocessing(file_paths):
    setup()
    result = start_preprocessing(file_paths)
    return result
