#pragma once
#include <seal/seal.h>

seal::Ciphertext
L2sqr_encrypted(seal::Evaluator &evaluator, seal::BatchEncoder &encoder,
                seal::Plaintext pt_vec, seal::Ciphertext ct_vec,
                seal::RelinKeys relin_keys, seal::GaloisKeys galois_keys,
                size_t dimensions);
