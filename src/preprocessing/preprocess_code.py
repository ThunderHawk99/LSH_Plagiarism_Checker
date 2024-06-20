from src.preprocessing.ProcessClass import CodePreprocessor
from src.constants import *
from src.common import *


def preprocess(data, language):
    preprocess_class = CodePreprocessor(data, language)
    return preprocess_class.process()


def paraphrase_token(token):
    if token.type in TOKEN_TYPES:
        token.value = token.type
    return token


def paraphrase(preprocessed_tokens, full_file_name):
    # Remove file extension
    file_name = full_file_name.split(".")[0]

    # Target file path
    target_paraphrased_directory_path = PARAPHRASED_DIR + f"{file_name}"
    create_dir_if_not_exists(target_paraphrased_directory_path)

    # Paraphrase the data 5 different times for 5 different results
    for i in range(PARAPHRASE_COUNT):

        # TODO - Paraphrase the data
        for token in preprocessed_tokens:
            token = paraphrase_token(token)

        tokens = [token.value for token in preprocessed_tokens]
        target_file_path_code = (
            f"{target_paraphrased_directory_path}/{file_name}_{PARAPHRASED}_{i}.txt"
        )
        write_to_file(target_file_path_code, " ".join(tokens))


def rebuild_code(tokens):
    code = ""
    indent_level = 0
    indent_spaces = "    "  # Assuming 4 spaces per indent level

    for token in tokens:
        if token.type == "whitespace":
            code += token.value
        elif token.type == "left brace":
            code += f" {token.value}\n"
            indent_level += 1
            code += indent_spaces * indent_level
        elif token.type == "right brace":
            indent_level -= 1
            code += f"\n{indent_spaces * indent_level}{token.value}"
        elif token.type == "semicolon":
            code += f"{token.value}\n{indent_spaces * indent_level}"
        else:
            code += f"{token.value}"

    return code


def save_preprocessed_data(preprocessed_tokens, target_file_path):
    tokens = [str(token) for token in preprocessed_tokens]
    code_target_file_path = target_file_path.replace(
        get_file_extension(target_file_path), ".txt"
    )
    write_to_file(code_target_file_path, "\n".join(tokens))


def process_single_code_file(file_path, language, data, full_file_name):
    # Perform preprocessing steps
    preprocessed_tokenised_data = preprocess(data, language)

    # Get the target file path
    target_file_path = file_path.replace(DATASET_DIR, PREPROCESSED_DIR)

    preprocessed_tokens = [token for token in preprocessed_tokenised_data]

    # Save the preprocessed data
    save_preprocessed_data(preprocessed_tokens, target_file_path)

    # Paraphrase the data
    paraphrase(preprocessed_tokens, full_file_name)

    return data
