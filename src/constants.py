from nltk.stem import WordNetLemmatizer
from multiprocessing import cpu_count
from nltk.corpus import stopwords
from nltk.data import find
import nltk

### Debug Constants
DEBUG = False
FRESH_RUN = True
WIKIPEDIA_DATA = True

PARAPHRASED = "paraphrased"

### Directory Constants
ASSETS_DIR = "assets/"
DATASET_DIR = ASSETS_DIR + "dataset/"
PREPROCESSED_DIR = ASSETS_DIR + "preprocessed/"
PROCESSED_DIR = ASSETS_DIR + "processed/"
PARAPHRASED_DIR = ASSETS_DIR + f"{PARAPHRASED}/"
EVALUATION_DIR = [ASSETS_DIR + "evaluation/"]
CORPUS_DIR = ASSETS_DIR + "pan-plagiarism-corpus-2011/"
DATASET_METADATA_DIR = ASSETS_DIR + "dataset_metadata/"
WIKIPEDIA_DIR = ASSETS_DIR + "wikipedia-dataset/"

### File Extensions
CSV_EXTENSION = ".csv"
JSON_EXTENSION = ".json"
PARQUET_EXTENSION = ".parquet"  #! Apache Parquet is a columnar storage file format available to any project in the Hadoop ecosystem
PNG_EXTENSION = ".png"
CODE_EXTENSIONS = {".py": "python", ".java": "java"}
TEXT_EXTENSIONS = [".txt"]
USED_FILE_EXTENSION = CSV_EXTENSION

### Code Token Types
TOKEN_TYPES = [
    "keyword",
    "class name",
    "function",
    "identifier",
    "string",
    "number",
    "keyword literal",
]

### corpus extraction
FILE_CUTOFF = 350
EXTRACT_ALL_FILES = False if FILE_CUTOFF > 0 else True

### File Names
KEYWORDS_FILE = PROCESSED_DIR + "keywords" + USED_FILE_EXTENSION
TEXTS_FILE = PROCESSED_DIR + "texts" + USED_FILE_EXTENSION
EVALUATION_IMG = "/evaluation" + PNG_EXTENSION
EVALUATION_CSV = "/evaluation" + CSV_EXTENSION
COMPLETE_EVALUATION_CSV = EVALUATION_DIR[0] + "complete_evaluation" + CSV_EXTENSION
FRAUD_PAIRS_FILE = PROCESSED_DIR + "fraud_pairs" + USED_FILE_EXTENSION

### Preprocessing Constants
WORD_BASED = [True]
KEYWORD_SELECTION_RATIO = [0.1]  # % How many keywords to select from each sentence
SUSPICIOUS_TERMS = ["suspicious", "SPUN"]
PARAPHRASE_COUNT = 1
PARAPHRASE_THRESHOLD = 0.2
RANDOM_SEED = 42


### Hashing Constants
N_HASH = [
    50,
    100,
    150,
    200,
    300,
    # 600,
]  # Number of hash functions, Increasing this will increase the probabibility of finding similar documents but also increase the complexity

### LSH Constants
N_BANDS = [
    # 5,
    10,
    25,
    50,
]  # Number of bands, increasing this will increase the probability of finding similar documents, we want to find a harmonious balance between
# N_BANDS and N_HASH as the rows depends on these two parameters. Increasing N_BANDS will decrease the number of rows and vice versa
# Higher number of rows require longer signatures to be similar
# Cross-ref check that N_HASH is a multiple of N_BANDS as we N_HASH[i] can be used with N_BANDS[j] with i != j
for band in N_BANDS:
    for hash_count in N_HASH:
        assert hash_count % band == 0, f"{hash_count} is not a multiple of {band}"

K = [
    # 1_000,
    # 5_000,
    # 10_000,
    20_000,
    50_000,
    100_000,
]  # Bucket size, a smaller bucket size increases the probability of finding similar documents but increases false positive rate.

SHINGLE_SIZE = [
    2,
    3,
    5,
    6,
    7,
    # 8,
    # 9,
]  # Shingle size, increasing this will increase the required size of matching patterns in documents to be considered similar.
# A shingle size of 5-7 is recommended as it catches most patterns.

WINDOW_STEP = [1, 2, 3, 5, 8]

OPTIMISATION_ITER_COUNT = (
    len(N_HASH) * len(N_BANDS) * len(K) * len(SHINGLE_SIZE) * len(WINDOW_STEP)
)

# Parameter for plotting results
TOP_VALUES = 10


### Multithreading Constants
# Source: https://stackoverflow.com/questions/20039659/python-multiprocessings-pool-process-limit
AMOUNT_OF_WORKERS = cpu_count() // 2


def download_nltk_resource(package_id, resource_name):
    try:
        # See if available
        find(resource_name)
    except Exception:
        # Not found so download the resource
        nltk.download(package_id, quiet=True)


# Download resources if not available
download_nltk_resource("stopwords", "corpora/stopwords")
download_nltk_resource("wordnet", "corpora/wordnet")
download_nltk_resource("punkt", "tokenizers/punkt")

set_of_stopwords = set(stopwords.words("english"))
lemmatiser = WordNetLemmatizer()
