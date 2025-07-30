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
    size_t m_CoarseProbe;
    size_t m_KNearest;

    size_t m_PreciseVectorDimensions;
    size_t m_Nlist;
    size_t m_Subquantizers;

    std::optional<Encryption> m_OptEncryption;

  public:
    explicit Client(size_t num_queries, size_t nprobe, size_t coarse_probe,
                    size_t k_nearest);

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

    // Returns NQUERY pairs of serialized ciphertexts and the relinearization
    // and galois keys, tuple.1 - vector of encrypted residual queries for
    // nqueries, tuple.2 - vector of encrypted residual squared lengths for
    // nqueries, tuple.3 - relinearization keys, tuple.4 - galois keys
    // tuple.5 - flat vec of encrypted query vectors
    std::tuple<std::vector<std::vector<std::vector<seal::seal_byte>>>,
               std::vector<std::vector<std::vector<seal::seal_byte>>>,
               std::vector<seal::seal_byte>, std::vector<seal::seal_byte>,
               std::vector<std::vector<seal::seal_byte>>>
    compute_encrypted_two_phase_search_parms(
        std::vector<float> &precise_queries, std::vector<float> &centroids,
        std::vector<faiss_idx_t> &nearest_centroids_idx) const;

    // Sends the residuals and nearest centroids to the server to perform a
    // coarse search and returns the encrypted coarse distances
    std::pair<std::vector<std::vector<std::vector<seal::seal_byte>>>,
              std::vector<std::vector<faiss_idx_t>>>
    get_encrypted_coarse_scores(
        const std::vector<std::vector<std::vector<seal::seal_byte>>>
            &serde_encrypted_residual_vecs,
        const std::vector<std::vector<std::vector<seal::seal_byte>>>
            &serde_encrypted_residual_vecs_squared,
        const std::vector<faiss_idx_t> &nprobe_nearest_centroids_idx,
        const std::vector<seal::seal_byte> &serde_relin_keys,
        const std::vector<seal::seal_byte> &serde_galois_keys) const;

    std::vector<std::vector<float>> deserialise_decrypt_coarse_distances(
        const std::vector<std::vector<std::vector<seal::seal_byte>>>
            &serde_encrypted_coarse_distances);

    // Sends the encrypted queries along with the nearest coarse vector ids to
    // the server to perform a precise search and return the encrypted precise
    // distances
    std::vector<std::vector<std::vector<seal::seal_byte>>> get_precise_scores(
        const std::vector<std::vector<seal::seal_byte>> &serde_precise_queries,
        const std::vector<std::vector<faiss_idx_t>> &nearest_coarse_vectors_id,
        const std::vector<seal::seal_byte> &serde_relin_keys,
        const std::vector<seal::seal_byte> &serde_galois_keys) const;

    std::vector<std::vector<float>> deserialise_decrypt_precise_distances(
        const std::vector<std::vector<std::vector<seal::seal_byte>>>
            &serde_encrypted_precise_distances);

    // void get_precise_vectors_pir(
    //     const std::array<std::array<DistanceIndexData, COARSE_PROBE>, NQUERY>
    //         &nearest_precise_vectors,
    //     std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>,
    //     K>,
    //                NQUERY> &query_results,
    //     std::array<std::array<faiss_idx_t, K>, NQUERY> &query_results_idx);
    //

    void benchmark_results(const std::vector<std::vector<faiss_idx_t>>
                               &k_nearest_vector_ids) const;

    // Selects the top `select_nearest_probe` vectors and returns their ids
    std::vector<std::vector<faiss_idx_t>> compute_nearest_vectors_id(
        const std::vector<std::vector<float>> &decrypted_distance_scores,
        const std::vector<std::vector<faiss_idx_t>> &vector_labels,
        const size_t num_queries, const size_t select_nearest_probe) const;

    // --------------------------------------
    // Single Phase Search

    // Returns NQUERY pairs of serialized ciphertexts and the relinearization
    // and galois keys, tuple.1 - vector of encrypted queries for
    // nqueries, tuple.2 - relinearization keys, tuple.3 - galois keys
    std::tuple<std::vector<std::vector<seal::seal_byte>>,
               std::vector<seal::seal_byte>, std::vector<seal::seal_byte>>
    compute_encrypted_single_phase_search_parms(
        std::vector<float> &precise_queries) const;

    // Sends the encrypted queries and nearest centroids to the server to
    // perform a single phase search and returns the encrypted precise distances
    std::pair<std::vector<std::vector<std::vector<seal::seal_byte>>>,
              std::vector<std::vector<faiss_idx_t>>>
    get_encrypted_single_phase_search_scores(
        const std::vector<std::vector<seal::seal_byte>>
            &serde_encrypted_queries,
        const std::vector<faiss_idx_t> &nprobe_nearest_centroids_idx,
        const std::vector<seal::seal_byte> &serde_relin_keys,
        const std::vector<seal::seal_byte> &serde_galois_keys) const;
};
