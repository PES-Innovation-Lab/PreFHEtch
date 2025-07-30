#pragma once

#include <memory>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <seal/seal.h>

#include "client_server_utils.h"

// Singleton class pattern for static access across all controllers
class Server {
  public:
    const size_t Nlist = 256;
    const size_t SubQuantizers = 32;
    const size_t SubQuantizerSize = 8;

  private:
    size_t m_PreciseVectorDimensions;
    size_t m_NBase;

    faiss::IndexFlatL2 m_Quantizer;
    std::unique_ptr<faiss::IndexIVFPQ> m_Index;
    std::vector<float> m_DatasetBase;

    size_t m_PolyModulusDegree;
    size_t m_PlaintextModulusSize;
    seal::EncryptionParameters m_EncryptionParms;

  public:
    Server();
    static std::shared_ptr<Server> &getInstance() {
        static std::shared_ptr<Server> server = std::make_shared<Server>();
        return server;
    }

    void init_index();
    static void run_webserver();

    // Returns NLIST centroids of PRECISE_DIMENSIONS each
    std::vector<float> retrieve_centroids() const;
    std::vector<seal::seal_byte> serialise_parms() const;

    std::tuple<std::vector<std::vector<seal::Ciphertext>>,
               std::vector<std::vector<seal::Ciphertext>>, seal::RelinKeys,
               seal::GaloisKeys>
    deserialise_coarse_search_parms(
        const std::vector<std::vector<std::vector<seal::seal_byte>>>
            &serde_encrypted_residual_vectors,
        const std::vector<std::vector<std::vector<seal::seal_byte>>>
            &serde_encrypted_residual_vectors_squared,
        const std::vector<seal::seal_byte> &serde_relin_keys,
        const std::vector<seal::seal_byte> &serde_galois_keys,
        // TODO: remove secret key, used for debugging
        const std::vector<seal::seal_byte> &sk) const;

    std::pair<std::vector<std::vector<seal::Ciphertext>>,
              std::vector<std::vector<faiss::idx_t>>>
    coarseSearch(
        std::vector<faiss::idx_t> &nprobe_centroids,
        std::vector<std::vector<seal::Ciphertext>> &encrypted_residual_queries,
        std::vector<std::vector<seal::Ciphertext>>
            &encrypted_residual_queries_squared,
        size_t num_queries, size_t nprobe, seal::RelinKeys &relin_keys,
        seal::GaloisKeys &galois_keys) const;

    std::tuple<std::vector<seal::Ciphertext>, seal::RelinKeys, seal::GaloisKeys>
    deserialise_precise_search_params(
        const std::vector<std::vector<seal::seal_byte>>
            &serde_encrypted_precise_queries,
        const std::vector<seal::seal_byte> &serde_relin_keys,
        const std::vector<seal::seal_byte> &serde_galois_keys) const;

    // takes nquery vectors with each query containing `coarse_probe`
    // number of encrypted queries, returns nquery vectors with each
    // vector containing a vector of `coarse_probe` number of distances
    std::vector<std::vector<seal::Ciphertext>> preciseSearch(
        const std::vector<std::vector<faiss::idx_t>> &nearest_coarse_vectors_id,
        const std::vector<seal::Ciphertext> &encrypted_precise_queries,
        const seal::RelinKeys &relin_keys,
        const seal::GaloisKeys &galois_keys) const;

    // void preciseVectorPIR(
    //     const std::array<std::array<faiss_idx_t, K>, NQUERY>
    //         &k_nearest_precise_vectors_idx,
    //     std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>,
    //     K>,
    //                NQUERY> &query_results);

    // helper for debugging
    void
    display_nprobe_centroids(const std::vector<faiss::idx_t> &nprobe_centroids,
                             const size_t num_queries) const;

    std::vector<std::vector<std::vector<seal::seal_byte>>>
    serialise_encrypted_distances(
        const std::vector<std::vector<seal::Ciphertext>> &encrypted_distances)
        const;

    // -----------------------------------
    // Single Phase Search

    std::tuple<std::vector<seal::Ciphertext>, seal::RelinKeys, seal::GaloisKeys>
    deserialise_single_phase_search_parms(
        const std::vector<std::vector<seal::seal_byte>>
            &serde_encrypted_query_vectors,
        const std::vector<seal::seal_byte> &serde_relin_keys,
        const std::vector<seal::seal_byte> &serde_galois_keys) const;

    std::pair<std::vector<std::vector<seal::Ciphertext>>,
              std::vector<std::vector<faiss::idx_t>>>
    singlePhaseSearch(const std::vector<faiss::idx_t> &nprobe_centroids,
                      const std::vector<seal::Ciphertext> &encrypted_queries,
                      const size_t num_queries, const size_t nprobe,
                      const seal::RelinKeys &relin_keys,
                      const seal::GaloisKeys &galois_keys) const;
};
