import numpy as np
import gc
from scipy.sparse import csr_matrix
from typing import List, Set, Tuple
from src.helpers.helper import timeit


# @timeit
# Creates shingles from a list of documents. These shingles are created by splitting the documents into words.
# This function gets called when WORD_BASED[0] = True in constants.py. Therefore these documents still consist of many words.
# @return a sparse matrix which encodes the shingles present in each document
def word_based_shingle(
    documents: List[str], shingle_size: int, window_step: int = 1
) -> csr_matrix:

    # Create a set for the unique shingles
    unique_shingles = set()

    # Iterate through each document
    for doc in documents:
        # Split the document into words
        words = doc.split()
        # Iterate through the words using the specified shingle size and window step in constants.py
        for i in range(0, len(words) - shingle_size + 1, window_step):
            # Create a shingle (tuple of words) of the specified size
            shingle = tuple(words[i : i + shingle_size])
            # Add the shingle to the set of unique shingles
            unique_shingles.add(shingle)

    # Create a sparse matrix of the shingles which encodes which shingles are present in which documents
    shingles = word_based_encode_shingles(
        documents, unique_shingles, shingle_size, window_step
    )

    return shingles


# Create a sparse matrix of the shingles which encodes which shingles are present in which documents
# This function encodes the shingles for the corresponding shingling operation, namely the word-based one
# @return a sparse matrix which encodes the shingles present in each document
def word_based_encode_shingles(
    documents: List[str],
    unique_shingles: Set[Tuple[str, ...]],
    shingle_size: int,
    window_step: int,
) -> csr_matrix:
    num_unique_shingles = len(unique_shingles)
    num_documents = len(documents)

    # Create a dictionary for quick lookup
    shingle_index = {shingle: idx for idx, shingle in enumerate(unique_shingles)}

    # Initialize lists to build the sparse matrix
    data = []
    rows = []
    cols = []

    # Iterate through each document and create a set of all its shingles (tuples of words)
    for doc_id, doc in enumerate(documents):
        words = doc.split()
        document_shingles = {
            tuple(words[i : i + shingle_size])
            for i in range(0, len(words) - shingle_size + 1, window_step)
        }
        # For each shingle in the document, encode it in the matrix with the corresponding shingle index and document index
        for shingle in document_shingles:
            idx = shingle_index.get(shingle)
            if idx is not None:
                data.append(1)
                rows.append(idx)
                cols.append(doc_id)

        # Perform garbage collection periodically to reduce performance and memory overhead
        if doc_id % 100 == 0:
            gc.collect()

    # Create a sparse matrix in Compressed Sparse Row format as shingles are mostly zeros for most documents
    shingles_matrix = csr_matrix(
        (data, (rows, cols)), shape=(num_unique_shingles, num_documents), dtype=np.uint8
    )  # use uint8 to save memory

    return shingles_matrix


# Creates shingles from a list of documents. These shingles are created by splitting the documents into character sequences of a specified size
# This function gets called when WORD_BASED[0] = False in constants.py. Therefore these documents consist of one long character string.
# @return a sparse matrix which encodes the shingles present in each document
@timeit
def character_based_shingle(
    documents: list, shingle_size: int, window_step: int = 1
) -> csr_matrix:
    # Create a set for the unique shingles
    unique_shingles = set()

    # Iterate through each document
    for doc in documents:
        # Iterate through the characters using the specified shingle size and window step
        for i in range(0, len(doc) - shingle_size + 1, window_step):
            # Create a k-gram shingle of the specified size
            shingle = doc[i : i + shingle_size]
            # Add the shingle to the set of unique shingles
            unique_shingles.add(shingle)

            del shingle
        gc.collect()

    shingles = char_based_encode_shingles(
        documents, unique_shingles, shingle_size, window_step=window_step
    )

    del documents
    del unique_shingles
    del shingle_size

    gc.collect()

    return shingles


# Create a sparse matrix of the shingles which encodes which shingles are present in which documents
# This function encodes the shingles for the corresponding shingling operation, namely the character-based one
# @return a sparse matrix which encodes the shingles present in each document
def char_based_encode_shingles(
    documents: list, unique_shingles: set, shingle_size: int, window_step: int
) -> csr_matrix:
    num_unique_shingles = len(unique_shingles)
    num_documents = len(documents)

    # Create a dictionary for quick lookup
    shingle_index = {shingle: idx for idx, shingle in enumerate(unique_shingles)}

    # Initialize lists to build the sparse matrix
    data = []
    rows = []
    cols = []

    # Iterate through each document and create a set of all its shingles
    for doc_id, doc in enumerate(documents):
        document_shingles = {
            doc[i : i + shingle_size]
            for i in range(0, len(doc) - shingle_size + 1, window_step)
        }
        # For each shingle in the document, encode it in the matrix with the corresponding shingle index and document index
        for shingle in document_shingles:
            idx = shingle_index.get(shingle)
            if idx is not None:
                data.append(1)
                rows.append(idx)
                cols.append(doc_id)

        # Perform garbage collection periodically to reduce performance overhead
        if doc_id % 100 == 0:
            gc.collect()

    # Create a sparse matrix in Compressed Sparse Row format as shingles are mostly zeros for most documents
    shingles_matrix = csr_matrix(
        (data, (rows, cols)), shape=(num_unique_shingles, num_documents), dtype=np.uint8
    )  # use uint8 to save memory

    return shingles_matrix
