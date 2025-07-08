#pragma once

#include <memory>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>

#include "client_server_utils.h"

constexpr int SERVER_PORT = 8080;

// Singleton class pattern for static access across all controllers
class Server {
  private:
    faiss::IndexFlatL2 m_Quantizer;
    faiss::IndexIVFPQ m_Index;
    std::vector<float> m_DatasetBase;

  public:
    Server();
    static std::shared_ptr<Server> &getInstance() {
        static std::shared_ptr<Server> srvr = std::make_shared<Server>();
        return srvr;
    }

    void init_index();
    void run_webserver();
    void retrieve_centroids(
        std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> &centroids);
    void coarseSearch(
        const std::array<float, PRECISE_VECTOR_DIMENSIONS> &precise_query,
        std::array<faiss::idx_t, NPROBE> &nearest_centroid_idx,
        std::vector<float> &coarse_distance_scores,
        std::vector<faiss::idx_t> &coarse_distance_indexes,
        std::array<size_t, NQUERY> &list_sizes_per_query);
    void preciseSearch(
        const std::array<float, PRECISE_VECTOR_DIMENSIONS> &precise_query,
        const std::array<faiss::idx_t, PRECISE_PROBE>
            &nearest_coarse_vector_idx,
        std::array<std::array<float, PRECISE_PROBE>, NQUERY>
            &precise_distance_scores);
    void preciseVectorPIR(
        const std::array<std::array<faiss_idx_t, K>, NQUERY>
            k_nearest_precise_vectors_idx,
        std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
                   NQUERY>
            query_results);
};
