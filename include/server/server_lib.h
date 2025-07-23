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

    // void Server::coarseSearch(
    //     std::vector<float> precise_queries,
    //     std::vector<faiss::idx_t> nearest_centroids,
    //     std::vector<float> &coarse_distance_scores,
    //     std::vector<faiss::idx_t> &coarse_distance_indexes,
    //     std::array<size_t, NQUERY> &list_sizes_per_query) const;
    //
    void preciseSearch(
        const std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, NQUERY>
            &precise_query,
        const std::array<std::array<faiss::idx_t, COARSE_PROBE>, NQUERY>
            &nearest_coarse_vector_idx,
        std::array<std::array<float, COARSE_PROBE>, NQUERY>
            &precise_distance_scores) const;

    void preciseVectorPIR(
        const std::array<std::array<faiss_idx_t, K>, NQUERY>
            &k_nearest_precise_vectors_idx,
        std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
                   NQUERY> &query_results);

    // Temp
    std::vector<faiss::idx_t>
    decrypt_centroids(std::vector<seal::seal_byte> &,
                      std::vector<seal::seal_byte> &) const;
    std::vector<float>
    decrypt_subvectors(std::vector<std::vector<seal::seal_byte>> &,
                       std::vector<std::vector<seal::seal_byte>> &,
                       std::vector<seal::seal_byte> &, size_t) const;
};
