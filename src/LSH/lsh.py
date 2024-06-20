import numpy as np
from src.LSH.hashing import Hashing
from src.helpers.helper import timeit


# Search for similar pairs of documents using the Locality Sensitive Hashing (LSH) algorithm
# @return the pairs of similar documents
# @timeit
def lsh(sig: np.ndarray, hashing: Hashing):
    # We will separate the signature matrix into bands and hash each band
    # Get the number of hash functions, bands, and documents
    nr_hash = sig.shape[0]
    nr_bands = hashing.n_bands
    nr_docs = sig.shape[1]
    # Calculate the number of rows per band
    r = nr_hash // nr_bands

    # Initialize a set to store the similar pairs
    similar_pairs = set()

    # Iterate through each band
    for i in range(nr_bands):
        band_start = i * r
        band_end = band_start + r
        # Get the band signatures for each document in the current band
        band_signatures = sig[band_start:band_end, :]

        # Compute hash for each column (document) in the current band by implementing numpy's apply_along_axis function
        hash_indices = np.apply_along_axis(
            hashing.hash_band_signature, 0, band_signatures, i
        )

        # Find candidate pairs by comparing the hash values of the signatures (of each document) in the current band
        # If the hash values are the same in one or more bands, the documents are considered similar
        for c1 in range(nr_docs):
            for c2 in range(c1 + 1, nr_docs):
                if hash_indices[c1] == hash_indices[c2]:
                    similar_pairs.add((c1, c2))

    return similar_pairs
