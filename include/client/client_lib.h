#pragma once

#include <vector>

#include <seal/seal.h>

#include "client_server_utils.h"

const std::string server_addr = "http://localhost:8080/";

struct DistanceIndexData {
    float distance;
    faiss_idx_t idx;
};

class Encryption {
  public:
    seal::EncryptionParameters EncryptedParms;
    seal::KeyGenerator KeyGen;
    // Use secret key mode to improve efficiency
    seal::SecretKey SecretKey;
    seal::Serializable<seal::RelinKeys> SerRelinKeys;

    seal::Encryptor Encryptor;
    seal::Decryptor Decryptor;
    seal::BatchEncoder BatchEncoder;

    Encryption(seal::EncryptionParameters encrypt_parms,
               const seal::SEALContext &context);
};

class Client {
  private:
    size_t m_PreciseVectorDimensions;
    size_t m_NumQueries;

    size_t m_Nlist;
    size_t m_Subquantizers;

    std::optional<Encryption> m_OptEncryption;

  public:
    explicit Client(size_t num_queries);

    std::vector<float> get_query();
    void encrypt_query();

    // Returns NLIST * NQUERY centroids
    std::pair<std::vector<float>, std::vector<seal::seal_byte>>
    get_centroids_encrypted_parms();

    void init_client_encrypt_parms(
        const std::vector<seal::seal_byte> &serde_encrypt_parms);

    // Return sorted NLIST centroids for each of the NQUERY queries
    std::vector<faiss_idx_t>
    sort_nearest_centroids(std::vector<float> &precise_queries,
                           std::vector<float> &centroids) const;

    // Returns a vector (NQUERY x SUBQUANTIZER) of pairs of serialized
    // ciphertexts
    // Pair.1 - encrypted precise subvector
    // Pair.2 - encrypted subvector squared length
    std::vector<
        std::pair<std::vector<seal::seal_byte>, std::vector<seal::seal_byte>>>
    compute_encrypted_subvector_components(
        std::vector<float> &precise_queries) const;

    void get_encrypted_coarse_scores(
        const std::vector<std::vector<faiss_idx_t>>
            &computed_nearest_centroids_idx,
        const std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, NQUERY>
            &precise_query,
        std::vector<float> &coarse_scores,
        std::vector<faiss_idx_t> &coarse_vectors_idx,
        std::array<size_t, NQUERY> &list_sizes_per_query_coarse);

    void compute_nearest_coarse_vectors(
        const std::vector<float> &coarse_distance_scores,
        const std::vector<faiss_idx_t> &coarse_vector_indexes,
        const std::array<size_t, NQUERY> &list_sizes_per_query_coarse,
        std::array<std::vector<DistanceIndexData>, NQUERY>
            &nearest_coarse_vectors_idx);

    void get_precise_scores(
        const std::array<std::vector<DistanceIndexData>, NQUERY>
            &sorted_coarse_vectors,
        const std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, NQUERY>
            &precise_query,
        std::array<std::array<float, COARSE_PROBE>, NQUERY> &precise_scores);

    void compute_nearest_precise_vectors(
        const std::array<std::array<float, COARSE_PROBE>, NQUERY>
            &precise_scores,
        // Uses the same index order for precise scores
        const std::array<std::vector<DistanceIndexData>, NQUERY>
            &sorted_coarse_vectors,
        std::array<std::array<DistanceIndexData, COARSE_PROBE>, NQUERY>
            &nearest_precise_vectors);

    void get_precise_vectors_pir(
        const std::array<std::array<DistanceIndexData, COARSE_PROBE>, NQUERY>
            &nearest_precise_vectors,
        std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
                   NQUERY> &query_results,
        std::array<std::array<faiss_idx_t, K>, NQUERY> &query_results_idx);

    void benchmark_results(const std::array<std::array<faiss_idx_t, K>, NQUERY>
                               &query_results_idx);
};
