#include "server_utils.h"

seal::Ciphertext
L2sqr_encrypted(seal::Evaluator &evaluator, seal::BatchEncoder &encoder,
                seal::Plaintext pt_vec, seal::Ciphertext ct_vec,
                seal::RelinKeys relin_keys, seal::GaloisKeys galois_keys,
                size_t dimensions) {

    seal::Ciphertext sub_result;
    evaluator.sub_plain(ct_vec, pt_vec, sub_result);

    seal::Ciphertext result;
    evaluator.square(sub_result, result);
    evaluator.relinearize_inplace(result, relin_keys);

    for (size_t step = 1; step < dimensions; step <<= 1) {
        seal::Ciphertext rotated;
        evaluator.rotate_rows(result, step, galois_keys, rotated);
        evaluator.add_inplace(result, rotated);
    }

    return result;
}
