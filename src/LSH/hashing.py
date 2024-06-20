import numpy as np
from src.constants import RANDOM_SEED


# Hashing class to generate hash functions and hash band signatures
class Hashing:
    def __init__(self, n_hash, n_bands: int = 4, K: int = 100000, seed=RANDOM_SEED):
        self._mersenne_prime = np.uint64((1 << 61) - 1)
        self._max_hash = np.uint64((1 << 32) - 1)
        self.seed = seed
        self.n_hash = n_hash
        self.n_bands = n_bands
        self.K = K

        rng = np.random.default_rng(self.seed)  # Local RNG instance
        # Generate random coefficients for MinHash functions
        self.As = rng.integers(
            1, self._mersenne_prime, size=(self.n_hash,), dtype=np.uint64
        )
        self.Bs = rng.integers(
            0, self._mersenne_prime, size=(self.n_hash,), dtype=np.uint64
        )
        # Generate random coefficients for LSH hashing
        self.coeff = rng.integers(
            1, self.K, size=(self.n_bands, self.n_hash // self.n_bands)
        )

    # One-pass-implementation of MinHashing. Hashes the index of the shingle to a hash value.
    # We implement a linear formula and use the modulo operation to avoid overflow
    def hash_idxs(self, index):
        return np.bitwise_and(
            (self.As * index + self.Bs) % self._mersenne_prime, self._max_hash
        ).astype(
            np.uint32
        )  # Change to uint32 to avoid memory overflow

    # Hashes the signature of a band to a hash value by using the dot product of random coefficients and the signature
    # Modulo operation is used to avoid overflow, K is the size of a bucket.
    def hash_band_signature(self, sig, i):
        return np.dot(self.coeff[i], sig) % self.K
