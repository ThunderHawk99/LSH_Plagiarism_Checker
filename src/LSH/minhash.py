import numpy as np
from src.LSH.hashing import Hashing
from scipy.sparse import csr_matrix
from src.helpers.helper import timeit


# Reduce the dimensionality of the shingles matrix by creating a signature matrix
# @return the signature matrix
# @timeit
def compute_signature_matrix(shingles: csr_matrix, hashing: Hashing) -> np.ndarray:
    num_documents = shingles.shape[1]
    # Initialize the signature matrix with max hash values. The dimensions are n_hash x num_documents
    sig = np.full(
        (hashing.n_hash, num_documents), hashing._max_hash, dtype=np.uint32
    )  # use uint32 to save memory

    # Iterate over each shingle
    for s_index in range(shingles.shape[0]):
        # Get the indices of documents that contain the shingle
        docs_with_shingle_indices = shingles[s_index].indices

        # Only proceed if there are documents with this shingle
        if docs_with_shingle_indices.size > 0:
            # Get the hash values for the index of the current shingle
            hashes = hashing.hash_idxs(s_index)

            # Update signature matrix using broadcasting
            # The minimum hash value is taken for each document that contains the shingle between the hash value and the current value in the signature matrix
            sig[:, docs_with_shingle_indices] = np.minimum(
                sig[:, docs_with_shingle_indices], hashes[:, np.newaxis]
            )

    # Once the computation is complete, return the signature matrix
    return sig
