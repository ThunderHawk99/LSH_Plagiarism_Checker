from sklearn.base import BaseEstimator
from src.LSH.shingle import word_based_shingle, character_based_shingle
from src.LSH.minhash import compute_signature_matrix
from src.LSH.lsh import lsh
from src.constants import N_HASH, N_BANDS, K, SHINGLE_SIZE, WINDOW_STEP, WORD_BASED
from src.LSH.hashing import Hashing
from src.helpers.helper import filter_pairs_by_number, index_to_filepath
import gc


# Create a model for LSH. This model is used to predict similar documents
class LSHModel(BaseEstimator):
    def __init__(
        self,
        shingle_size=SHINGLE_SIZE,
        window_step=WINDOW_STEP,
        n_bands=N_BANDS,
        K=K,
        n_hash=N_HASH,
    ):
        self.shingle_size = shingle_size
        self.window_step = window_step
        self.K = K
        self.hashing = Hashing(n_hash=n_hash, n_bands=n_bands, K=K)

    # Given a set of documents, predict the similar documents
    # @return the pairs of similar documents
    def predict(self, X, file_paths=None, optimization=False):
        shingles = None
        # Compute shingles with the correct shingling operation, depending on whether we want to operate with word- or character-shingles.
        if WORD_BASED[0]:
            shingles = word_based_shingle(documents=X, shingle_size=self.shingle_size)
        else:
            shingles = character_based_shingle(
                documents=X, shingle_size=self.shingle_size
            )

        del X
        # Compute signature matrix
        signature_matrix = compute_signature_matrix(
            shingles=shingles, hashing=self.hashing
        )

        del shingles
        # Apply LSH algorithm
        candidate_pairs = lsh(sig=signature_matrix, hashing=self.hashing)

        del signature_matrix
        del self

        # Filter out pairs where both elements represent the same file.
        # This means we do not allow files to be compared with themselves or (paraphrased) versions of themselves
        candidate_pairs = filter_pairs_by_number(candidate_pairs, file_paths)

        # If we are not optimizing (aka making final prediction), convert the indices in the candidate pairs to file paths for readability
        if not optimization:
            candidate_pairs = index_to_filepath(candidate_pairs, file_paths)

        del file_paths
        gc.collect()

        return candidate_pairs

    # Score the model by comparing the predicted pairs with the ground truth. We use F1 score to evaluate the model
    # @return the F1 score, precision, recall, false positives, false negatives, true positives, true negatives
    def score(self, X, y, file_paths=None):
        # Get the predicted pairs from the LSH model
        predicted_pairs = self.predict(X, file_paths=file_paths, optimization=True)

        del file_paths

        # Calculate true positives, false positives, and false negatives using their definitions
        true_positives = len(predicted_pairs & y)

        # Calculate false positives & negatives using set difference
        false_positives = len(predicted_pairs - y)
        false_negatives = len(y - predicted_pairs)

        # Calculate true negatives
        total_possible_pairs = len(X) * (len(X) - 1) // 2
        true_negatives = total_possible_pairs - (
            true_positives + false_positives + false_negatives
        )

        del X
        gc.collect()

        # Add a small constant to precision and recall to avoid division by zero
        epsilon = 1e-9

        # Calculate precision
        precision = (true_positives + epsilon) / (
            true_positives + false_positives + epsilon
        )
        # Calculate recall
        recall = (true_positives + epsilon) / (
            true_positives + false_negatives + epsilon
        )

        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall)

        return (
            f1_score,
            precision,
            recall,
            false_positives,
            false_negatives,
            true_positives,
            true_negatives,
        )
