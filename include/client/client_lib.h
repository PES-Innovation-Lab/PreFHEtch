#pragma once

#include <vector>

#include <seal/seal.h>

#include "client_server_utils.h"

const std::string server_addr = "http://localhost:8080/";

class Encryption {
  public:
    seal::EncryptionParameters EncryptedParms;
    seal::KeyGenerator KeyGen;
    // Use secret key mode to improve efficiency
    seal::SecretKey SecretKey;
    seal::Serializable<seal::RelinKeys> SerdeRelinKeys;
    seal::Serializable<seal::GaloisKeys> SerdeGaloisKeys;

    seal::Encryptor Encryptor;
    seal::Decryptor Decryptor;
    seal::BatchEncoder BatchEncoder;

    Encryption(seal::EncryptionParameters encrypt_parms,
               const seal::SEALContext &context);
};

class Client {
  private:
    size_t m_NumQueries;
    size_t m_NProbe;

    size_t m_PreciseVectorDimensions;
    size_t m_Nlist;
    size_t m_Subquantizers;

    std::optional<Encryption> m_OptEncryption;

  public:
    explicit Client(size_t num_queries, size_t nprobe);

    std::vector<float> get_query();
    void encrypt_query();

    // Returns NLIST * NQUERY centroids
    std::pair<std::vector<float>, std::vector<seal::seal_byte>>
    get_centroids_encrypted_parms();

    void init_client_encrypt_parms(
        const std::vector<seal::seal_byte> &serde_encrypt_parms);

    // Return sorted NLIST centroids for each of the NQUERY queries and sorted
    // NPROBE centroids for each of the NQUERIES
    std::pair<std::vector<faiss_idx_t>, std::vector<faiss_idx_t>>
    sort_nearest_centroids(std::vector<float> &precise_queries,
                           std::vector<float> &centroids) const;

    // Returns NQUERY pairs of serialize ciphertexts and the relinearization
    // and galois keys

    // tuple.1 - vector of encrypted residual queries for nqueries

    // tuple.2 - vector of encrypted residual squared lengths for nqueries

    // tuple.3 - relinearization keys

    // tuple.4 - galois keys
    std::tuple<std::vector<std::vector<std::vector<seal::seal_byte>>>,
               std::vector<std::vector<std::vector<seal::seal_byte>>>,
               std::vector<seal::seal_byte>, std::vector<seal::seal_byte>>
    compute_encrypted_coarse_search_parms(
        std::vector<float> &precise_queries, std::vector<float> &centroids,
        std::vector<faiss_idx_t> &nearest_centroids_idx) const;

    std::pair<std::vector<std::vector<std::vector<seal::seal_byte>>>,
              std::vector<std::vector<faiss_idx_t>>>
    get_encrypted_coarse_scores(
        std::vector<std::vector<std::vector<seal::seal_byte>>>
            &serde_encrypted_vecs,
        std::vector<std::vector<std::vector<seal::seal_byte>>>
            &serde_encrypted_vecs_squared,
        std::vector<faiss_idx_t> &nprobe_nearest_centroids_idx,
        std::vector<seal::seal_byte> &serde_relin_keys,
        std::vector<seal::seal_byte> &serde_galois_keys) const;

    std::vector<std::vector<float>> deserialise_decrypt_coarse_distances(
        const std::vector<std::vector<std::vector<seal::seal_byte>>>
            &serde_encrypted_coarse_distances);

    std::vector<std::vector<faiss_idx_t>> compute_nearest_coarse_vectors_idx(
        const std::vector<std::vector<float>> &decrypted_coarse_distance_scores,
        const std::vector<std::vector<faiss_idx_t>> &coarse_vector_labels,
        const size_t num_queries, const size_t coarse_probe) const;

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
