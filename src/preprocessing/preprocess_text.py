from src.preprocessing.ProcessClass import TextPreprocessor
from nltk.corpus import wordnet
from collections import defaultdict
from src.constants import *
from src.common import *
import random
from rake_nltk import Rake

# Use seed for reproducibility
random.seed(RANDOM_SEED)


# Custom Rake Wrapper to allow for custom keyword selection ratio
class CustomRake(Rake):
    def __init__(self, set_of_stopwords=None):
        super().__init__(stopwords=set_of_stopwords)

    def process_text(self, text):
        # If {KEYWORD_SELECTION_RATIO} is 1, then we just add each word from the sentence as a keyword
        # in original order.
        if KEYWORD_SELECTION_RATIO[0] == 1:
            return nltk.word_tokenize(text)

        # Split text into sentences
        sentences = self._tokenize_text_to_sentences(text)
        document_keywords = []

        # Go through each sentence
        for sentence in sentences:
            self.extract_keywords_from_text(sentence)
            keywords_with_scores = self.get_ranked_phrases_with_scores()

            # Get only the top {KEYWORD_SELECTION_RATIO}% of the keywords
            # NEW: Keep in mind to use at least one keyword
            num_keywords = max(
                1, int(len(keywords_with_scores) * KEYWORD_SELECTION_RATIO[0])
            )
            top_keywords = keywords_with_scores[:num_keywords]

            # Add the keywords to the document keywords
            document_keywords.extend(keyword for _, keyword in top_keywords)

        return document_keywords


def preprocess(data):
    # Initialize the Rake object
    rake_object = CustomRake()

    preprocess_class = TextPreprocessor(data, set_of_stopwords, lemmatiser, rake_object)
    return preprocess_class.process()


# Cache the synonyms to avoid repeated calls to the WordNet API
synonyms_cache = defaultdict(list)


# Get synonyms for a word
def get_synonyms(word):
    synonyms = {
        lemma.name().replace("_", " ")
        for syn in wordnet.synsets(word)
        for lemma in syn.lemmas()
        if "_" not in (synonym := lemma.name().replace("_", " ")) and synonym != word
    }
    return list(synonyms)


# Paraphrase the keywords
def paraphrase_keywords(words):
    transformed_keywords = []

    for keyword in words:
        if random.random() >= PARAPHRASE_THRESHOLD:
            transformed_keywords.append(keyword)
        else:
            synonyms = synonyms_cache[keyword]
            if not synonyms:
                synonyms = get_synonyms(keyword)
                synonyms_cache[keyword] = synonyms
            synonym = random.choice(synonyms) if synonyms else keyword
            transformed_keywords.append(synonym.lower())

    return transformed_keywords


# Paraphrase the document
def paraphrase(data, full_file_name):
    # Remove file extension
    file_name = full_file_name.split(".")[0]

    # Target file path
    target_paraphrased_directory_path = PARAPHRASED_DIR + f"{file_name}"
    create_dir_if_not_exists(target_paraphrased_directory_path)

    # Initialize the results
    results = []

    # Paraphrase the data 5 different times for 5 different results
    for i in range(PARAPHRASE_COUNT):
        target_file_path = (
            f"{target_paraphrased_directory_path}/{file_name}_{PARAPHRASED}_{i}.txt"
        )
        paraphrased_data = None
        # If it already exists, read it in
        if os.path.exists(target_file_path):
            with open(target_file_path, "r", encoding="utf-8-sig") as file:
                paraphrased_data = file.read()
        else:
            paraphrased_data = paraphrase_keywords(data)
            paraphrased_data = convert_to_string(paraphrased_data)
            write_to_file(target_file_path, paraphrased_data)

        results.append((paraphrased_data, target_file_path))

    return results


# Convert the data to either words or one long string
def convert_to_string(paraphrased_data):
    if WORD_BASED[0]:
        return " ".join(paraphrased_data)
    else:
        s = ""
        for word in paraphrased_data:
            word = word.replace(" ", "")
            for char in word:
                s += char
        return s


# Process single file
def process_single_text_file(file_path, data, full_file_name):
    # Get the target file path
    target_file_path = file_path.replace(DATASET_DIR, PREPROCESSED_DIR)
    # Initialize the result
    preprocessed_tokenised_data = None

    # Check if the target_file_path exists, if so, load it in
    if os.path.exists(target_file_path):
        with open(target_file_path, "r", encoding="utf-8-sig") as file:
            result = file.read()
    else:
        # Perform preprocessing steps
        preprocessed_tokenised_data = preprocess(data)
        # Save the preprocessed data
        result = convert_to_string(preprocessed_tokenised_data)
        # Write the preprocessed data to the target file path
        write_to_file(target_file_path, result)

    paraphrased = []

    # Paraphrase the data if full_full_name contains a string from SUSPICIOUS_TERMS
    if any(term in full_file_name for term in SUSPICIOUS_TERMS):
        paraphrased = paraphrase(preprocessed_tokenised_data, full_file_name)

    return [(result, target_file_path)] + paraphrased
