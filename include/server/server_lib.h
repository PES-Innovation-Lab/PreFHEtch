#pragma once

#include <memory>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>

#include "client_server_utils.h"

// Singleton class pattern for static access across all controllers
class Server {
  private:
    faiss::IndexFlatL2 m_Quantizer;
    faiss::IndexIVFPQ m_Index;

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
    void prefilter(
        const std::array<float, PRECISE_VECTOR_DIMENSIONS> &precise_query,
        std::array<int64_t, NPROBE> &nearest_centroid_idx,
        std::vector<float> &coarse_distance_scores,
        std::vector<faiss::idx_t> &coarse_distance_indexes,
        std::array<size_t, NQUERY> &list_sizes_per_query);
};
